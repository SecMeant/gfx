#include <array>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <limits>
#include <optional>
#include <vector>
#include <cassert>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <vulkan/vulkan.h>

using layer_properties = std::vector<VkLayerProperties>;

[[nodiscard]] static layer_properties
get_layer_properties()
{
  uint32_t instance_layer_count;
  layer_properties props;
  VkResult res;

  /*
   * It's possible, though very rare, that the number of
   * instance layers could change. For example, installing something
   * could include new layers that the loader would pick up
   * between the initial query for the count and the
   * request for VkLayerProperties. The loader indicates that
   * by returning a VK_INCOMPLETE status and will update the
   * the count parameter.
   * The count parameter will be updated with the number of
   * entries loaded into the data pointer - in case the number
   * of layers went down or is smaller than the size given.
   */
  do {
    res = vkEnumerateInstanceLayerProperties(&instance_layer_count, NULL);
    if (res)
      break;

    if (instance_layer_count == 0)
      break;

    props.resize(instance_layer_count);
    res = vkEnumerateInstanceLayerProperties(&instance_layer_count, props.data());
  } while (res == VK_INCOMPLETE);

  return props;
}

using instance_extensions = std::vector<VkExtensionProperties>;

[[nodiscard]] static instance_extensions
enumerate_extensions()
{
  uint32_t prop_count;
  instance_extensions props;

  if (vkEnumerateInstanceExtensionProperties(nullptr, &prop_count, nullptr) != VK_SUCCESS) {
    printf("Failed to enumerate layer extensions\n");
    return props;
  }

  props.resize(prop_count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &prop_count, props.data()) != VK_SUCCESS) {
    printf("Failed to enumerate layer extensions\n");
    props.resize(0);
    return props;
  }

  return props;
}

[[nodiscard]] static std::expected<VkInstance, VkResult>
create_instance(const char *app_name, const layer_properties &layers, const instance_extensions &extensions)
{
  using std::unexpected;

  VkResult res;
  VkInstance instance;

  const VkApplicationInfo app_info = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pNext = nullptr,
    .pApplicationName = app_name,
    .applicationVersion = 1,
    .pEngineName = app_name,
    .engineVersion = 1,
    .apiVersion = VK_API_VERSION_1_0,
  };

  std::vector<const char*> layer_names;
  layer_names.emplace_back("VK_LAYER_KHRONOS_validation");

  std::vector<const char *> extension_names;
  for (const auto &e : extensions)
    extension_names.emplace_back(e.extensionName);

  const VkInstanceCreateInfo inst_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .pApplicationInfo = &app_info,
    .enabledLayerCount = static_cast<uint32_t>(layer_names.size()),
    .ppEnabledLayerNames = layer_names.size() ? layer_names.data() : nullptr,
    .enabledExtensionCount = static_cast<uint32_t>(extension_names.size()),
    .ppEnabledExtensionNames = extension_names.data()
  };

  res = vkCreateInstance(&inst_info, nullptr, &instance);
  if (res != VK_SUCCESS)
    return unexpected(res);

  return instance;
}

struct device_info
{
  VkPhysicalDevice physical_device;
  VkDevice device;
  uint32_t queue_family_index; // Index of compute family queue
};

[[nodiscard]] static std::expected<device_info, VkResult>
make_device(VkInstance instance)
{
  VkResult res;
  uint32_t dev_count;
  std::vector<VkPhysicalDevice> devs;
  device_info ret {
    .physical_device = {},
    .device = {},
    .queue_family_index = uint32_t(-1),
  };

  res = vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
  if (res != VK_SUCCESS)
    return std::unexpected(res);

  if (dev_count == 0)
    return std::unexpected(VkResult(-1));

  devs.resize(dev_count);

  res = vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());
  if (res != VK_SUCCESS) {
    printf("Failed to enumerate physical devices\n");
    return std::unexpected(res);
  }

  // FIXME: for now we just take first reported GPU, but we might want to change later
  const auto pdev = devs[0];
  ret.physical_device = pdev;

  uint32_t queue_properties_count;
  std::vector<VkQueueFamilyProperties> queue_properties;
  //VkPhysicalDeviceMemoryProperties memory_properties;
  //VkPhysicalDeviceProperties device_properties;

  vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_properties_count, nullptr);
  if (queue_properties_count == 0) {
    printf("WARNING: got number of queue family properties == 0 for device %p\n", pdev);
    return std::unexpected(VkResult(-1));
  }

  queue_properties.resize(queue_properties_count);
  vkGetPhysicalDeviceQueueFamilyProperties(pdev, &queue_properties_count, queue_properties.data());

  //vkGetPhysicalDeviceMemoryProperties(pdev, &memory_properties);
  //vkGetPhysicalDeviceProperties(pdev, &device_properties);

  for (auto i = 0u; i < queue_properties.size(); ++i) {

    const auto &prop = queue_properties[i];
    if (prop.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      ret.queue_family_index = i;
      break;
    }

  }

  if (ret.queue_family_index == uint32_t(-1)) {
    printf("Failed to find queue family with compute bit enabled\n");
    return std::unexpected(VkResult(-1));
  }

  const float queue_priorities[] = { 1.0f };

  const VkDeviceQueueCreateInfo queue_create_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .queueFamilyIndex = ret.queue_family_index,
    .queueCount = 1,
    .pQueuePriorities = queue_priorities,
  };

  const VkDeviceCreateInfo dev_create_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_create_info,
    .enabledLayerCount = 0,
    .ppEnabledLayerNames = nullptr,
    .enabledExtensionCount = 0,
    .ppEnabledExtensionNames = nullptr,
    .pEnabledFeatures = nullptr,
  };

  res = vkCreateDevice(pdev, &dev_create_info, nullptr, &ret.device);
  if (res != VK_SUCCESS) {
    printf("Failed to create logical device: %d\n", res);
  }

  return ret;
}

struct memory_info
{
  VkBuffer input_buffer;
  VkBuffer output_buffer;
  VkDeviceMemory input_memory;
  VkDeviceMemory output_memory;
};

static constexpr VkDeviceSize INPUT_BUFFER_SIZE_BYTES = 1024;
static constexpr VkDeviceSize OUTPUT_BUFFER_SIZE_BYTES = 1024;

[[nodisacrd]] static std::expected<memory_info, VkResult>
allocate_memory(const device_info device_info)
{
  VkResult res;
  memory_info ret;

  /*
   * Create input and output buffer.
   */

  const VkBufferCreateInfo input_buffer_create_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .size  = INPUT_BUFFER_SIZE_BYTES,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = 1,
    .pQueueFamilyIndices = &device_info.queue_family_index,
  };

  res = vkCreateBuffer(device_info.device, &input_buffer_create_info, nullptr, &ret.input_buffer);
  if (res != VK_SUCCESS) {
    printf("Failed to create input buffer: %d\n", res);
    return std::unexpected(VkResult(-1));
  }

  const VkBufferCreateInfo output_buffer_create_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .size  = OUTPUT_BUFFER_SIZE_BYTES,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = 1,
    .pQueueFamilyIndices = &device_info.queue_family_index,
  };

  res = vkCreateBuffer(device_info.device, &output_buffer_create_info, nullptr, &ret.output_buffer);
  if (res != VK_SUCCESS) {
    printf("Failed to create output buffer: %d\n", res);
    return std::unexpected(VkResult(-1));
  }


  /*
   * Query memory types and search for memory that is device local.
   */

  VkPhysicalDeviceMemoryProperties memory_properties;
  vkGetPhysicalDeviceMemoryProperties(device_info.physical_device, &memory_properties);

  const auto opt_device_mem_index = [&] () -> std::optional<uint32_t> {

    for (auto i = 0u; i < memory_properties.memoryTypeCount; ++i) {
      const auto mtype = memory_properties.memoryTypes[i];

      if (mtype.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        return i;

    }

    return std::nullopt;

  }();

  if (!opt_device_mem_index.has_value()) {
    printf("Failed to find device local memory\n");
    return std::unexpected(VkResult(-1));
  }

  const VkMemoryAllocateInfo memory_allocation_input_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .pNext = nullptr,
    .allocationSize = INPUT_BUFFER_SIZE_BYTES,
    .memoryTypeIndex = opt_device_mem_index.value(),
  };

  res = vkAllocateMemory(device_info.device, &memory_allocation_input_info, nullptr, &ret.input_memory);
  if (res != VK_SUCCESS) {
    printf("Failed to allocate memory for input: %d\n", res);
    return std::unexpected(VkResult(-1));
  }

  const VkMemoryAllocateInfo memory_allocation_output_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .pNext = nullptr,
    .allocationSize = OUTPUT_BUFFER_SIZE_BYTES,
    .memoryTypeIndex = opt_device_mem_index.value(),
  };

  res = vkAllocateMemory(device_info.device, &memory_allocation_output_info, nullptr, &ret.output_memory);
  if (res != VK_SUCCESS) {
    printf("Failed to allocate memory for output: %d\n", res);
    return std::unexpected(VkResult(-1));
  }

  res = vkBindBufferMemory(device_info.device, ret.input_buffer, ret.input_memory, 0);
  if (res != VK_SUCCESS) {
    printf("Failed to bind input memory to a buffer: %d\n", res);
    return std::unexpected(VkResult(-1));
  }

  res = vkBindBufferMemory(device_info.device, ret.output_buffer, ret.output_memory, 0);
  if (res != VK_SUCCESS) {
    printf("Failed to bind output memory to a buffer: %d\n", res);
    return std::unexpected(VkResult(-1));
  }

  return ret;
}

struct shader_data
{
  uint8_t *data;
  uint32_t size;
};

static std::optional<shader_data> read_shader_(const char * const filepath)
{
  int shader_fd = -1;
  struct stat statbuf = {};
  struct shader_data ret = {};

  shader_fd = open(filepath, 0, O_RDONLY);

  if (shader_fd == -1) {
    perror("shader");
    return std::nullopt;
  }

  struct defer_fd_close_t {
    defer_fd_close_t(int fd)
      : fd(fd) {}

    ~defer_fd_close_t()
    { close(this->fd); }

    int fd;

  } const defer_fd_close(shader_fd);

  if (fstat(shader_fd, &statbuf)) {
    perror("fstat");
    return std::nullopt;
  }

  /*
   * We allocate with operator new, so we can just leak it and let the kernel
   * do the cleaning.
   */
  ret.data = new uint8_t[statbuf.st_size];
  ret.size = statbuf.st_size;

  if (!ret.data) {
    printf("Failed to allocate memory for shader\n");
    return std::nullopt;
  }

  /*
   * FIXME: We probably want to read in loop until we read entire file, instead
   *        of assuming that a single read would suffice.
   */
  if (read(shader_fd, ret.data, ret.size) != ret.size) {
    printf("Failed to read shader data\n");
    return std::nullopt;
  }

  return ret;
}

struct shader_info
{
  VkShaderModule module;
  struct shader_data data;
};

static std::expected<shader_info, VkResult> load_shaders(const device_info device_info)
{
  VkResult res;
  VkShaderModule shader_module;

  auto shader_data = read_shader_("square.spv").value();

  const VkShaderModuleCreateInfo shader_module_create_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .codeSize = shader_data.size,
    .pCode = reinterpret_cast<const uint32_t*>(shader_data.data),
  };

  res = vkCreateShaderModule(device_info.device, &shader_module_create_info, nullptr, &shader_module);
  if (res != VK_SUCCESS) {
    printf("Failed to create shader module: %d\n", res);
    return std::unexpected(res);
  }

  return shader_info { .module = shader_module, .data = shader_data };
}

struct pipeline_info
{
  VkDescriptorSetLayout descriptor_set_layout;
  VkPipelineLayout layout;
  VkPipeline pipeline;
};

static std::expected<pipeline_info, VkResult>
configure_pipeline(const device_info device_info, const shader_info shader_info)
{
  VkResult res;
  VkDescriptorSetLayout descriptor_set_layout;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;

  const VkDescriptorSetLayoutBinding descriptor_set_layout_bindings[2] = {
    [0] = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = nullptr,
    },

    [1] = {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = nullptr,
    },
  };

  const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0, // FIXME: should the flags be 0? Not sure.
    .bindingCount = 2,
    .pBindings = descriptor_set_layout_bindings,
  };

  res = vkCreateDescriptorSetLayout(device_info.device,
      &descriptor_set_layout_create_info, nullptr, &descriptor_set_layout);
  if (res != VK_SUCCESS) {
    printf("Failed to create descriptor set layout: %d\n", res);
    return std::unexpected(res);
  }

  const VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .setLayoutCount = 1,
    .pSetLayouts = &descriptor_set_layout,

    /*
     * TODO: Experiment with push constants. For example add bias that is added
     *       after squaring to each element.
     */
    .pushConstantRangeCount = 0,
    .pPushConstantRanges = nullptr,
  };

  res = vkCreatePipelineLayout(device_info.device,
      &pipeline_layout_create_info, nullptr, &pipeline_layout);
  if (res != VK_SUCCESS) {
    printf("Failed to create pipeline layout: %d\n", res);
    return std::unexpected(res);
  }

  const VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = shader_info.module,
    .pName = "main",
    .pSpecializationInfo = nullptr, // FIXME: Should it be nullptr?
  };

  const VkComputePipelineCreateInfo compute_pipeline_create_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .stage = pipeline_shader_stage_create_info,
    .layout = pipeline_layout,
    .basePipelineHandle = VK_NULL_HANDLE,
    .basePipelineIndex = 0,
  };

  res = vkCreateComputePipelines(device_info.device, VK_NULL_HANDLE,
      1, &compute_pipeline_create_info, nullptr, &pipeline);
  if (res != VK_SUCCESS) {
    printf("Failed to create pipeline: %d\n", res);
    return std::unexpected(res);
  }

  return pipeline_info {
    .descriptor_set_layout = descriptor_set_layout,
    .layout = pipeline_layout,
    .pipeline = pipeline,
  };
}

struct descriptor_info
{
  VkDescriptorPool descriptor_pool;
  VkDescriptorSet descriptor_set;
};

static std::expected<descriptor_info, VkResult>
create_descriptors(const device_info device_info, const pipeline_info pipeline_info, const memory_info memory_info)
{
  VkResult res;
  VkDescriptorPool descriptor_pool;
  VkDescriptorSet descriptor_set;

  const VkDescriptorPoolSize descriptor_pool_sizes[1] = {
    [0] = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 2,
    },
  };

  const VkDescriptorPoolCreateInfo descriptor_pool_create_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .maxSets = 2,
    .poolSizeCount = 1,
    .pPoolSizes = descriptor_pool_sizes,
  };

  res = vkCreateDescriptorPool(device_info.device, &descriptor_pool_create_info,
      nullptr, &descriptor_pool);
  if (res != VK_SUCCESS) {
    printf("Failed to allocate descriptor pool: %d\n", res);
    return std::unexpected(res);
  }

  const VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .pNext = nullptr,
    .descriptorPool = descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &pipeline_info.descriptor_set_layout,
  };

  res = vkAllocateDescriptorSets(device_info.device, &descriptor_set_allocate_info, &descriptor_set);
  if (res != VK_SUCCESS) {
    printf("Failed to allocate descriptor sets: %d\n", res);
    return std::unexpected(res);
  }

  const VkDescriptorBufferInfo input_buffer_info = {
    .buffer = memory_info.input_buffer,
    .offset = 0,
    .range = INPUT_BUFFER_SIZE_BYTES,
  };

  const VkDescriptorBufferInfo output_buffer_info = {
    .buffer = memory_info.output_buffer,
    .offset = 0,
    .range = OUTPUT_BUFFER_SIZE_BYTES,
  };

  const VkWriteDescriptorSet write_descriptor_sets[2] = {
    [0] = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .dstArrayElement = 0, // FIXME: is 0 ok here?
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pImageInfo = nullptr,
      .pBufferInfo = &input_buffer_info,
      .pTexelBufferView = nullptr,
    },

    [1] = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = descriptor_set,
      .dstBinding = 1,
      .dstArrayElement = 0, // FIXME: is 0 ok here?
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pImageInfo = nullptr,
      .pBufferInfo = &output_buffer_info,
      .pTexelBufferView = nullptr,
    },
  };

  vkUpdateDescriptorSets(device_info.device, 2, write_descriptor_sets, 0, nullptr);

  return descriptor_info {
    .descriptor_pool = descriptor_pool,
    .descriptor_set = descriptor_set,
  };
}

int main(void)
{
  const auto props = get_layer_properties();

#if defined(OPT_PRINT_VULKAN_LAYERS)

  printf("Found %zu layers\n", props.size());
  for (const auto &p : props)
    printf("\t%s: %s\n", p.layerName, p.description);

#endif

  const auto extensions = enumerate_extensions();

#if defined(OPT_PRINT_VULKAN_EXTENSIONS)

  printf("Found %zu extensions\n", extensions.size());
  for (const auto &e : extensions)
    printf("\t%s: %u\n", e.extensionName, e.specVersion);

#endif

  const auto instance = create_instance("vkexplore", props, extensions).value();
  const auto device_info = make_device(instance).value();
  const auto memory_info = allocate_memory(device_info).value();
  const auto shader_info = load_shaders(device_info).value();
  const auto pipeline_info = configure_pipeline(device_info, shader_info).value();
  const auto descriptor_info = create_descriptors(device_info, pipeline_info, memory_info).value();

  return 0;
}
