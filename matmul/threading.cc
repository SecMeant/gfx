#include "threading.h"

#include <fmt/format.h>

/*
 * Implementation of the main idle loop.
 *
 * The code could have been much cleaner if we got rid of the verbose logs,
 * but sometimes they are useful, so...
 */
void thread_pool::idle(const u32 thread_id, struct work_context * const wctx)
{
    /* TODO: pin to CPU. */

    if constexpr (CONFIG_THREAD_TRACE)
        fmt::print(stderr, "{}: th{}\n", __func__, thread_id);

    while (1) {

        /* lock guard */ {

            if constexpr (CONFIG_THREAD_TRACE)
                fmt::print(stderr, "{}: th{} idling\n", __func__, thread_id);

            /* 
             * TODO: All threads will be block on that mutex when wait return.
             *       We don't really want that, can we make it faster?
             */
            std::unique_lock lck(wctx->m);
            wctx->cv_submitted.wait(lck, [thread_id, wctx]{ return wctx->bits_pending.test(thread_id); });

            if (!wctx->work) {
                if constexpr (CONFIG_THREAD_TRACE)
                    fmt::print(stderr, "{}: th{} exiting\n", __func__, thread_id);

                wctx->bits_pending.set(thread_id, false);
                return;
            }

        }

        if constexpr (CONFIG_THREAD_TRACE)
            fmt::print(stderr, "{}: th{} working\n", __func__, thread_id);

        wctx->work(thread_id);

        if constexpr (CONFIG_THREAD_TRACE)
            fmt::print(stderr, "{}: th{} work done\n", __func__, thread_id);

        bool iam_last = false;
        /* lock guard */ {

            std::unique_lock lck(wctx->m);

            wctx->bits_pending.set(thread_id, false);
            iam_last = wctx->bits_pending.none();

            if constexpr (CONFIG_THREAD_TRACE)
                fmt::print(stderr, "{}: th{} reporting {}\n", __func__, thread_id, wctx->bits_pending.to_string());

        }

        if (iam_last) {
            if constexpr (CONFIG_THREAD_TRACE)
                fmt::print(stderr, "{}: th{} notify\n", __func__, thread_id);

            wctx->cv_finished.notify_all();
        }
    }
}

void thread_pool::exit_threads()
{
    this->schedule(thread_pool::WorkType());

    for (auto &th: this->threads)
        if (th.joinable())
            th.join();
}

void thread_pool::schedule(const WorkType work)
{
    if (this->num_threads() == 0) [[unlikely]]
        return;

    /* lock guard */ {
        std::unique_lock lck(this->wctx.m);

        /* Wait for the previous work to finish. */
        this->wctx.cv_finished.wait(lck, [this]{ return this->wctx.bits_pending.none(); });

        this->wctx.work = std::move(work);
        this->wctx.bits_pending = this->wctx.submit_mask;
    }

    this->wctx.cv_submitted.notify_all();
}

void thread_pool::sync()
{
    if (this->num_threads() == 0) [[unlikely]]
        return;

    std::unique_lock lck(this->wctx.m);
    this->wctx.cv_finished.wait(lck, [this]{ return this->wctx.bits_pending.none(); });
}

void thread_pool::resize(const u32 num_threads)
{
    /*
     * This will may block for a while if the threads are working.
     * Keep that in mind.
     */
    this->exit_threads();

    this->wctx.work = thread_pool::WorkType();
    this->wctx.bits_pending.reset();
    this->wctx.submit_mask.reset();

    this->threads.clear();
    this->threads.reserve(num_threads);

    for (u32 thread_id = 0; thread_id < num_threads; ++thread_id) {
        this->wctx.submit_mask.set(thread_id);
        this->threads.emplace_back(std::thread(idle, thread_id, &this->wctx));
    }
}

