#pragma once

#include "VulkanDef.h"

namespace mvk::util
{
    constexpr std::array VALIDATION_LAYERS = {
        "VK_LAYER_KHRONOS_validation",
    };
    constexpr std::array DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    };

    // Instance
    std::vector<const char*> GetRequiredExtensions();

    bool CheckValidationLayerSupport();
    bool CheckInstanceExtensionSupport(const std::vector<const char*>& requiredExtensions);

    vk::DebugUtilsMessengerCreateInfoEXT PopulateDebugMessengerCreateInfo();
    vk::DebugUtilsMessengerEXT SetupDebugMessenger(vk::Instance instance);
    void DestroyDebugMessenger(vk::Instance instance, vk::DebugUtilsMessengerEXT debugMessenger);

    // Device
    bool IsDeviceSuitable(vk::PhysicalDevice device, vk::SurfaceKHR surface);
    bool CheckDeviceExtensionSupport(vk::PhysicalDevice device);
    bool CheckSwapChainAdequate(vk::PhysicalDevice device, vk::SurfaceKHR surface);

    QueueFamilyIndices FindQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface);
    GPUInfo EvaluatePhysicalDevice(vk::PhysicalDevice device, vk::SurfaceKHR surface);
    DeviceFeatures QueryDeviceFeatures(vk::PhysicalDevice device);
    vk::SampleCountFlagBits GetMaxUsableSampleCount(vk::PhysicalDevice device);

    uint32_t CalculateDeviceScore(const GPUInfo& info, vk::SurfaceKHR surface);
    uint64_t GetTotalDeviceMemory(const vk::PhysicalDeviceMemoryProperties& memProps);
    std::string GetDeviceTypeString(vk::PhysicalDeviceType type);

    bool IsVulkan13OrLater(vk::PhysicalDevice device);
    bool IsVulkan12OrLater(vk::PhysicalDevice device);
    bool IsVulkan11OrLater(vk::PhysicalDevice device);

    uint32_t GetVulkanVersion(vk::PhysicalDevice device);
    std::string GetVulkanVersionString(vk::PhysicalDevice device);
    std::tuple<uint32_t, uint32_t, uint32_t> GetVulkanVersionNumbers(vk::PhysicalDevice device);

    // Info
	void LogSystemInfo();
    void LogDeviceInfo(vk::PhysicalDevice device, const DeviceFeatures& features,const QueueFamilyIndices& queueFamilyIndices);
    void LogGPUSelectionInfo(const std::vector<GPUInfo>& gpuInfos, const GPUInfo& selected);

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
}