#include <cstdio>
#include <vector>
#include <expected>

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

[[nodiscard]] static std::expected<VkInstance, VkResult>
create_instance(const char *app_name)
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

  const VkInstanceCreateInfo inst_info = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .pApplicationInfo = &app_info,

    /* TODO: Pass detected and enabled layers and extensions */
    .enabledLayerCount = 0,
    .ppEnabledLayerNames = nullptr,
    .enabledExtensionCount = 0,
    .ppEnabledExtensionNames = nullptr
  };

  res = vkCreateInstance(&inst_info, nullptr, &instance);
  if (res != VK_SUCCESS)
    return unexpected(res);

  return instance;
}

[[nodiscard]] static std::vector<VkPhysicalDevice>
enumerate_devices(VkInstance instance)
{
  VkResult res;
  uint32_t dev_count;
  std::vector<VkPhysicalDevice> devs;

  vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
  if (dev_count == 0)
    return devs;

  devs.resize(dev_count);

  res = vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());

  return devs;
}

int main(void)
{
  const auto props = get_layer_properties();

#if defined(OPT_PRINT_VULKAN_LAYERS)

  printf("Found %zu layers\n", props.size());
  for (const auto &p : props)
    printf("\t%s: %s\n", p.layerName, p.description);

#endif

  const auto instance = create_instance("vkexplore").value();

  return 0;
}
