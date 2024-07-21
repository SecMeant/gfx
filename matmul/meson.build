project('matmul', 'cpp')

fmt = dependency('fmt', required: true)
mipc = dependency('mipc', required: true)
rapidjson = dependency('RapidJSON', required: true)

test_files_path = meson.source_root() + '/tests/files/'

cpp_args_common = [
  '-std=c++23',
  '-Wall',
  '-Wextra',
]

libmatmul_src = [
    'matmul_cpu_naive.cc',
    'random.cc',
    'threading.cc',
]

libmatmul = static_library(
  'matmul',
  libmatmul_src,
  cpp_args: cpp_args_common,
  dependencies: [
    fmt,
    mipc,
    rapidjson,
  ],
)

python3 = find_program('python3')

get_type_name_h = custom_target(
    'get_type_name.h',
    output : 'get_type_name.h',
    input : 'srcgen/gen_get_type_str',
    command : [python3, '@INPUT@', '@OUTPUT@'],
)

executable(
    'bench',
    ['benchmark.cc'],
    link_with: libmatmul,
    cpp_args: cpp_args_common,
)

test_src = [
  'tests/test.cc',
  'tests/threading_test.cc',
  'tests/test_against_pytorch.cc',
  get_type_name_h,
]

executable(
  'test_matmul',
  test_src,
  cpp_args: [
    cpp_args_common,
    '-DCONFIG_TEST_FILES_PATH="' + test_files_path + '"',
  ],
  link_with: libmatmul,
  include_directories: ['../']
)
