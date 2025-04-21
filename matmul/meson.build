project(
  'matmul',
  ['cpp', 'cuda'],
  meson_version: '>=1.1'
)

conf = configuration_data()
summary_info = {}

fmt = dependency('fmt', required: true)
mipc = dependency('mipc', required: true)
rapidjson = dependency('RapidJSON', required: true)
opencl = dependency('OpenCL', required: true)
cuda = dependency('cuda', required: true)

test_files_path = meson.project_source_root() + '/tests/files/'
grad_file_path  = meson.project_source_root() + '/tests/files/grad.safetensors'
classify_file_path  = meson.project_source_root() + '/tests/files/classify.safetensors'

cpp_args_common = [
  '-std=c++23',
  '-Wall',
  '-Wextra',

  #
  # Clang complains about passing plain, "runtime",
  # strings as the format and only parameter to the printf.
  #
  # Just shut it's mouth. Clang's just wrong.
  #
  '-Wno-format-security',

  '-DCL_TARGET_OPENCL_VERSION=300',
]

cuda_args = [
  '--dopt=on',
  '--optimize', '3',
  '-Wno-deprecated-gpu-targets',
]

conf.set_quoted('CONFIG_CL_MATMUL_KERNEL_SRC', meson.project_source_root() / 'matmul.cl')

foreach key : conf.keys()
  summary_info += {key: conf.get(key)}
endforeach

config_header = configure_file(output: 'config.h', configuration: conf)

libmatmul_src = [
    'matmul_cpu_naive.cc',
    'matmul_opencl.cc',
    'random.cc',
    'threading.cc',
    'matmul_cuda.cu',
    'matmul_cuda.cc',
]

libmatmul = static_library(
    'matmul',
    libmatmul_src,
    config_header,
    cpp_args: cpp_args_common,
    cuda_args: cuda_args,
    dependencies: [
        fmt,
        mipc,
        rapidjson,
        opencl,
        cuda,
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
    '-DCONFIG_GRAD_FILE_PATH="' + grad_file_path + '"',
    '-DCONFIG_CLASSIFY_FILE_PATH="' + classify_file_path + '"',
  ],
  link_with: libmatmul,
  include_directories: ['../']
)
