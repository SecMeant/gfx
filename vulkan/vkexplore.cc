#include <array>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <limits>
#include <vector>
#include <cassert>

#include <unistd.h>

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

  res = vkCreateBuffer(device_info.device, &output_buffer_create_info, nullptr, &ret.input_buffer);
  if (res != VK_SUCCESS) {
    printf("Failed to create output buffer: %d\n", res);
    return std::unexpected(VkResult(-1));
  }


  /*
   * Query memory types and search for memory that is host and device visible.
   */

  VkPhysicalDeviceMemoryProperties memory_properties;
  vkGetPhysicalDeviceMemoryProperties(device_info.physical_device, &memory_properties);

  return ret;
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

  return 0;
}
