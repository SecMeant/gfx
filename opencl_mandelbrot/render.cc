#include "config.h"

#if CONFIG_RENDER_TO_FILE
static void
bitmap_save(u8* const bitmap, const u32 width, const u32 height)
{
    using cv::Mat;
    using cv::imwrite;

    Mat bitmap_mat(height, width, CV_8UC4, bitmap);

    imwrite(CONFIG_RENDER_OUTPUT_FILE_NAME, bitmap_mat);
}
#else
static void
bitmap_save(u8* const bitmap, const u32 width, const u32 height)
{
    using cv::imshow;
    using cv::Mat;
    using cv::waitKey;

    Mat bitmap_mat(height, width, CV_8UC4, bitmap);

    imshow("Bitmap Image", bitmap_mat);

    while (waitKey(0) != 'q') {
    }
}
#endif

