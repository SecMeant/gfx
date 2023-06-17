#include <array>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <limits>
#include <vector>
#include <cassert>

#include <unistd.h>

#define VK_USE_PLATFORM_XCB_KHR
#include <vulkan/vulkan.h>

#include <xcb/xcb.h>

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
  for (const auto &p : layers)
    layer_names.emplace_back(p.layerName);

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
  VkPhysicalDevice device;
  std::vector<VkQueueFamilyProperties> queue_properties;
  VkPhysicalDeviceMemoryProperties memory_properties;
  VkPhysicalDeviceProperties device_properties;
};

[[nodiscard]] static std::vector<device_info>
enumerate_devices(VkInstance instance)
{
  VkResult res;
  uint32_t dev_count;
  std::vector<VkPhysicalDevice> devs;
  std::vector<device_info> ret;

  vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
  if (dev_count == 0)
    return ret;

  devs.resize(dev_count);

  res = vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());
  if (res != VK_SUCCESS) {
    printf("Failed to enumerate physical devices\n");
    return ret;
  }

  for (const auto &dev: devs) {
    uint32_t queue_properties_count;
    std::vector<VkQueueFamilyProperties> queue_properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkPhysicalDeviceProperties device_properties;

    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_properties_count, nullptr);
    if (queue_properties_count == 0) {
      printf("WARNING: got number of queue family properties == 0 for device %p\n", dev);
      continue;
    }

    queue_properties.resize(queue_properties_count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_properties_count, queue_properties.data());

    vkGetPhysicalDeviceMemoryProperties(dev, &memory_properties);
    vkGetPhysicalDeviceProperties(dev, &device_properties);

    ret.emplace_back(device_info {
      .device = dev,
      .queue_properties = std::move(queue_properties),
      .memory_properties = memory_properties,
      .device_properties = device_properties,
    });
  }

  return ret;
}

struct xcb_connection_context
{
  xcb_connection_t *connection;
  xcb_screen_t *screen;
};

static std::expected<xcb_connection_context, int>
init_xcb_connection()
{
  const xcb_setup_t *setup;
  xcb_screen_iterator_t iter;
  int scr;

  xcb_connection_context xctx;

  xctx.connection = xcb_connect(NULL, &scr);
  if (xctx.connection == NULL || xcb_connection_has_error(xctx.connection)) {
    return std::unexpected(1);
  }

  setup = xcb_get_setup(xctx.connection);
  iter = xcb_setup_roots_iterator(setup);
  while (scr-- > 0)
    xcb_screen_next(&iter);
  xctx.screen = iter.data;

  return xctx;
}

struct xcb_window_context
{
  xcb_window_t window;
  xcb_intern_atom_reply_t *atom_wm_delete_window;
};

static std::expected<xcb_window_context, int>
init_xcb_window(xcb_connection_context &cctx)
{
  uint32_t value_mask, value_list[32];
  xcb_window_context wctx;

  wctx.window = xcb_generate_id(cctx.connection);

  value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
  value_list[0] = cctx.screen->black_pixel;
  value_list[1] = XCB_EVENT_MASK_KEY_RELEASE | XCB_EVENT_MASK_EXPOSURE;

  xcb_create_window(cctx.connection, XCB_COPY_FROM_PARENT, wctx.window, cctx.screen->root, 0, 0, 800, 600, 0,
      XCB_WINDOW_CLASS_INPUT_OUTPUT, cctx.screen->root_visual, value_mask, value_list);

  xcb_intern_atom_cookie_t cookie = xcb_intern_atom(cctx.connection, 1, 12, "WM_PROTOCOLS");
  xcb_intern_atom_reply_t *reply = xcb_intern_atom_reply(cctx.connection, cookie, 0);

  xcb_intern_atom_cookie_t cookie2 = xcb_intern_atom(cctx.connection, 0, 16, "WM_DELETE_WINDOW");
  wctx.atom_wm_delete_window = xcb_intern_atom_reply(cctx.connection, cookie2, 0);

  xcb_change_property(cctx.connection, XCB_PROP_MODE_REPLACE, wctx.window, reply->atom, 4, 32, 1,
      &(*wctx.atom_wm_delete_window).atom);

  free(reply);

  xcb_map_window(cctx.connection, wctx.window);

  const uint32_t coords[] = {100, 100};
  xcb_configure_window(cctx.connection, wctx.window, XCB_CONFIG_WINDOW_X | XCB_CONFIG_WINDOW_Y, coords);
  xcb_flush(cctx.connection);

  xcb_generic_event_t *e;
  while ((e = xcb_wait_for_event(cctx.connection))) {
    if ((e->response_type & ~0x80) == XCB_EXPOSE)
      break;
  }

  return wctx;
}

struct surface_context
{
  VkSurfaceKHR surface;
  uint32_t graphics_queue_family_index;
  uint32_t present_queue_family_index;
};

static std::expected<surface_context, VkResult>
init_swapchain_extension(
    VkInstance instance,
    const device_info &pdev,
    const xcb_connection_context &xcb_cctx,
    const xcb_window_context &xcb_wctx
) {
  VkResult res;
  VkSurfaceKHR surface;

  VkXcbSurfaceCreateInfoKHR create_info = {
    .sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
    .pNext = NULL,
    .flags = 0,
    .connection = xcb_cctx.connection,
    .window = xcb_wctx.window,
  };

  res = vkCreateXcbSurfaceKHR(instance, &create_info, NULL, &surface);
  if (res != VK_SUCCESS) {
    printf("Failed to create xcb surface: %d\n", res);
    return std::unexpected(res);
  }

  std::vector<VkBool32> supports_present;
  supports_present.resize(pdev.queue_properties.size());
  for (uint32_t i = 0; i < pdev.queue_properties.size(); ++i) {
    vkGetPhysicalDeviceSurfaceSupportKHR(pdev.device, i, surface, &supports_present[i]);
  }

  constexpr auto BAD_FAMILY_INDEX = std::numeric_limits<uint32_t>::max();
  uint32_t graphics_queue_family_index = BAD_FAMILY_INDEX;
  uint32_t present_queue_family_index = BAD_FAMILY_INDEX;
  for (uint32_t i = 0; i < pdev.queue_properties.size(); ++i) {
    if (pdev.queue_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      if (graphics_queue_family_index == BAD_FAMILY_INDEX)
        graphics_queue_family_index = i;

      if (supports_present[i] == VK_TRUE) {
        graphics_queue_family_index = i;
        present_queue_family_index  = i;
        break;
      }
    }
  }

  /* If we haven't found queue that supports both graphics and present, then
   * find a separate present queue. */
  if (present_queue_family_index == BAD_FAMILY_INDEX) {
    for (size_t i = 0; i < pdev.queue_properties.size(); ++i) {
      if (supports_present[i] == VK_TRUE) {
        present_queue_family_index = i;
      }
    }
  }

  if (graphics_queue_family_index == BAD_FAMILY_INDEX ||
      present_queue_family_index == BAD_FAMILY_INDEX) {
    printf("Couldn't find graphics and present queues.");
    return std::unexpected(VK_INCOMPLETE); // TODO: we probably shouldn't return VK_INCOMPLETE here
  }

  if (graphics_queue_family_index != present_queue_family_index) {
    printf("Graphics and present queues are not the same. We don't support it for now.");
    return std::unexpected(VK_INCOMPLETE); // TODO: we probably shouldn't return VK_INCOMPLETE here
  }

  return surface_context {
    .surface = surface,
    .graphics_queue_family_index = graphics_queue_family_index,
    .present_queue_family_index = present_queue_family_index,
  };
}

static std::expected<VkDevice, VkResult>
init_device(
    const device_info &devinfo,
    const surface_context &sctx,
    const layer_properties &layers,
    const instance_extensions &extensions
) {
  VkResult res;
  VkDevice device;
  std::array<float, 1> queue_priorities = {0.0};

  VkDeviceQueueCreateInfo queue_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .queueFamilyIndex = sctx.graphics_queue_family_index,
    .queueCount = queue_priorities.size(),
    .pQueuePriorities = queue_priorities.data(),
  };

  std::vector<const char*> layer_names;
  for (const auto &p : layers)
    layer_names.emplace_back(p.layerName);

  std::vector<const char *> extension_names;
  extension_names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  VkDeviceCreateInfo device_info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_info,
    .enabledLayerCount = static_cast<uint32_t>(layer_names.size()),
    .ppEnabledLayerNames = layer_names.data(),
    .enabledExtensionCount = static_cast<uint32_t>(extension_names.size()),
    .ppEnabledExtensionNames = extension_names.size() ? extension_names.data() : nullptr,
    .pEnabledFeatures = NULL, // TODO: should it be null?
  };

  res = vkCreateDevice(devinfo.device, &device_info, NULL, &device);
  if (res != VK_SUCCESS) {
    printf("Failed to create device: %i\n", res);
    return std::unexpected(res);
  }

  return device;
}

static void init_queues()
{

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

  const auto devices = enumerate_devices(instance);
  if (devices.size() == 0) {
    printf("No devices found");
    return 1;
  }

  printf("Found %zu devices\n", devices.size());

  auto xcb_cctx = init_xcb_connection().value();
  auto xcb_wctx = init_xcb_window(xcb_cctx).value();

  const auto &defpdev = devices[0];
  const auto sctx = init_swapchain_extension(instance, defpdev, xcb_cctx, xcb_wctx).value();
  const auto defdev = init_device(defpdev, sctx, props, extensions).value();

  sleep(2);

  return 0;
}
