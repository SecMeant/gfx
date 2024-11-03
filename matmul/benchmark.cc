#include "types.h"
#include "mat.h"
#include "print_utils.h"

static void parse_args(int argc, char **argv)
{
    (void) argc;
    (void) argv;
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    auto m = mat_i64_t::make_matrix(3, 4);

    print_mat(m);

    return 0;
}