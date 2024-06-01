#include "types.h"
#include "matview.h"
#include "mat.h"

void matmul_cpu_naive(matview_t lhs, matview_t rhs, matview_t out);

static void parse_args(int argc, char **argv)
{
    (void) argc;
    (void) argv;
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    return 0;
}