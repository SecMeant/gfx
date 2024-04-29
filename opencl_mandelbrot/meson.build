project('mandelbrot', 'cpp')

conf = configuration_data()
summary_info = {}

opencv_dep = dependency('opencv4')
opencl_dep = dependency('OpenCL')

cpp = meson.get_compiler('cpp')

sources = ['mandelbrot.cc']

if not get_option('CONFIG_CL_COMPILE_ONLINE')
  opencl_compile_llvm_command = [
    'clang',
    '-O3',
    '-c',
    '-target', 'spir64',
    '-emit-llvm',
    '-cl-std=CL3.0',
    '-o', '@OUTPUT@',
    '@INPUT@',
  ]

  opencl_mandelbrot_llvm_target = custom_target(
    'cl_mandelbrot_llvm',
    input: 'mandelbrot.cl',
    output: 'mandelbrot.bc',
    command: opencl_compile_llvm_command,
    build_always: true,
  )

  opencl_compile_spirv_command = [
    'llvm-spirv',
    '@INPUT@',
    '-o', '@OUTPUT@',
  ]

  opencl_mandelbrot_spirv_target = custom_target(
    'cl_mandelbrot_spriv',
    input: opencl_mandelbrot_llvm_target,
    output: 'mandelbrot.spv',
    command: opencl_compile_spirv_command,
    build_always: true,
  )
endif

conf.set('CONFIG_CL_COMPILE_ONLINE', get_option('CONFIG_CL_COMPILE_ONLINE').to_int().to_string())
conf.set('CONFIG_RENDER_TO_FILE', get_option('CONFIG_RENDER_TO_FILE').to_int().to_string())
conf.set_quoted('CONFIG_RENDER_OUTPUT_FILE_NAME', get_option('CONFIG_RENDER_OUTPUT_FILE_NAME'))

foreach conf_key : conf.keys()
  summary_info += {conf_key: conf.get(conf_key)}
endforeach

config_header = configure_file(output: 'config.h', configuration: conf)

executable(
    'mandelbrot',
    sources,
    config_header,
    dependencies: [
      opencv_dep,
      opencl_dep
    ],
    cpp_args: [
      '-std=c++17',
      '-Wall',
      '-Wextra',
      '-DCL_TARGET_OPENCL_VERSION=300',
    ]
)

summary(summary_info)
