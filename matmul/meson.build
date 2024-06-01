project('matmul', 'cpp')

cpp_args_common = [
  '-std=c++23',
  '-Wall',
  '-Wextra',
]

libmatmul_src = [
    'matmul_cpu_naive.cc',
    'random.cc',
]

libmatmul = static_library(
  'matmul',
  libmatmul_src,
  cpp_args: cpp_args_common,
)

executable(
    'bench',
    ['benchmark.cc'],
    link_with: libmatmul,
    cpp_args: cpp_args_common,
)

test_src = [
  'tests/test.cc',
]

executable(
  'test_matmul',
  test_src,
  cpp_args: cpp_args_common,
  link_with: libmatmul,
  include_directories: ['../']
)

