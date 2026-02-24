#include "stdafx.h"
#include "VulkanUtil.h"

std::vector<const char*> mvk::util::GetRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = nullptr;

    try
    {
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    }
    catch (...) {}

    std::vector<const char*> extensions;
    if (glfwExtensions && glfwExtensionCount > 0)
    {
        extensions = std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
    }
    else
    {
        extensions = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME
        };
    }

    if constexpr (ENABLE_VALIDATION_LAYERS)
    {
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    
    if (!util::CheckInstanceExtensionSupport(extensions))
    {
        throw std::runtime_error("Required instance extensions not supported!");
    }
    return extensions;
}

bool mvk::util::CheckValidationLayerSupport()
{
    try
    {
        const auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const auto& layerName : VALIDATION_LAYERS)
        {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers)
            {
                if (std::strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                mcore::Logger::Error("Validation layer not found: ", layerName);
                mcore::Logger::Print();
                return false;
            }
        }

        return true;
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error checking validation layer support: ", e.what());
        mcore::Logger::Print();
        return false;
    }
}

bool mvk::util::CheckInstanceExtensionSupport(const std::vector<const char*>& requiredExtensions)
{
    try
    {
        const auto availableExtensions = vk::enumerateInstanceExtensionProperties();

        for (const auto& required : requiredExtensions)
        {
            bool found = false;
            for (const auto& available : availableExtensions)
            {
                if (std::strcmp(required, available.extensionName) == 0)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                mcore::Logger::Error("Required instance extension not supported: ", required);
                mcore::Logger::Print();
                return false;
            }
        }

        return true;
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error checking instance extension support: ", e.what());
        mcore::Logger::Print();
        return false;
    }
}

vk::DebugUtilsMessengerCreateInfoEXT mvk::util::PopulateDebugMessengerCreateInfo()
{
    return vk::DebugUtilsMessengerCreateInfoEXT(
        {},
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        DebugCallback
    );
}

vk::DebugUtilsMessengerEXT mvk::util::SetupDebugMessenger(vk::Instance instance)
{
    if constexpr (!ENABLE_VALIDATION_LAYERS)
        return nullptr;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo = PopulateDebugMessengerCreateInfo();
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(instance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
    if (!func)
    {
        throw std::runtime_error("vkCreateDebugUtilsMessengerEXT function not found!");
    }

    VkDebugUtilsMessengerEXT debugMessenger;
    VkResult result = func(
        static_cast<VkInstance>(instance),
        reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo),
        nullptr,
        &debugMessenger
    );

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create debug messenger!");
    }

    return static_cast<vk::DebugUtilsMessengerEXT>(debugMessenger);
}

void mvk::util::DestroyDebugMessenger(vk::Instance instance, vk::DebugUtilsMessengerEXT debugMessenger)
{
    if (debugMessenger)
    {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
        if (!func)
        {
            throw std::runtime_error("vkDestroyDebugUtilsMessengerEXT function not found!");
        }

        func(
            static_cast<VkInstance>(instance),
            static_cast<VkDebugUtilsMessengerEXT>(debugMessenger),
            nullptr
        );
    }
}

bool mvk::util::IsDeviceSuitable(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    bool basicRequirements =
        FindQueueFamilies(device, surface).IsComplete() &&
        CheckDeviceExtensionSupport(device) &&
        CheckSwapChainAdequate(device, surface);
    if (!basicRequirements) return false;

    auto features = QueryDeviceFeatures(device);
    if (!features.dynamicRendering)
    {
        return false;
    }
    return true;
}

bool mvk::util::CheckDeviceExtensionSupport(vk::PhysicalDevice device)
{
    const auto availableExtensions = device.enumerateDeviceExtensionProperties();
    std::set<std::string_view> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

    const auto props = device.getProperties();
    const auto features = QueryDeviceFeatures(device);
    if (features.dynamicRendering && props.apiVersion < VK_API_VERSION_1_3)
    {
        requiredExtensions.insert(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        requiredExtensions.insert(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
        requiredExtensions.insert(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    }

    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    if (!requiredExtensions.empty())
    {
        mcore::Logger::Debug("Missing required extensions:");
        for (const auto& missing : requiredExtensions)
        {
            mcore::Logger::Info("  ", missing);
        }
        mcore::Logger::Print();
        return false;
    }
    return true;
}

bool mvk::util::CheckSwapChainAdequate(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    if (!surface) return true;

    const auto formats = device.getSurfaceFormatsKHR(surface);
    const auto presentModes = device.getSurfacePresentModesKHR(surface);
    return !formats.empty() && !presentModes.empty();
}

mvk::QueueFamilyIndices mvk::util::FindQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    QueueFamilyIndices indices;
    const auto queueFamilyProperties = device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
    {
        const auto& props = queueFamilyProperties[i];
        if (props.queueFlags & vk::QueueFlagBits::eGraphics)
        {
            indices.graphics = i;
        }
        if ((props.queueFlags & vk::QueueFlagBits::eCompute) &&
            !(props.queueFlags & vk::QueueFlagBits::eGraphics) &&
            !indices.compute.has_value())
        {
            indices.compute = i;
        }
        if ((props.queueFlags & vk::QueueFlagBits::eTransfer) &&
            !(props.queueFlags & vk::QueueFlagBits::eGraphics) &&
            !(props.queueFlags & vk::QueueFlagBits::eCompute) &&
            !indices.transfer.has_value())
        {
            indices.transfer = i;
        }

        if (surface)
        {
            if (device.getSurfaceSupportKHR(i, surface))
            {
                indices.present = i;
            }
        }
    }
    if (!indices.IsComplete())
    {
        throw std::runtime_error("No suitable queue families found!");
    }
    return indices;
}

mvk::GPUInfo mvk::util::EvaluatePhysicalDevice(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
    GPUInfo info{};
    info.device = device;
    info.properties = device.getProperties();
    info.features = device.getFeatures();
    info.memoryProperties = device.getMemoryProperties();

    if (!IsDeviceSuitable(device, surface))
    {
        info.score = 0;
        info.scoreBreakdown = "Failed basic requirements";
        return info;
    }

    info.supportedFeatures = QueryDeviceFeatures(device);
    info.score = CalculateDeviceScore(info, surface);

    return info;
}

mvk::DeviceFeatures mvk::util::QueryDeviceFeatures(vk::PhysicalDevice device)
{
    DeviceFeatures features{};

    uint32_t instanceVersion = 0;
    vkEnumerateInstanceVersion(&instanceVersion);

    const auto props = device.getProperties();
    const auto isVulkan13 = props.apiVersion >= VK_API_VERSION_1_3;
    const auto isVulkan12 = props.apiVersion >= VK_API_VERSION_1_2;

    if (isVulkan13)
    {
        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vk::PhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan13Features.pNext = &vulkan12Features;

        vk::PhysicalDeviceFeatures2 deviceFeatures{};
        deviceFeatures.pNext = &vulkan13Features;

        device.getFeatures2(&deviceFeatures);

        features.dynamicRendering = vulkan13Features.dynamicRendering;
        features.synchronization2 = vulkan13Features.synchronization2;
        features.maintenance4 = vulkan13Features.maintenance4;
        features.timelineSemaphore = vulkan12Features.timelineSemaphore;
        features.bufferDeviceAddress = vulkan12Features.bufferDeviceAddress;
        features.descriptorIndexing = vulkan12Features.descriptorIndexing;
    }
    else
    {
        const auto extensions = device.enumerateDeviceExtensionProperties();
        for (const auto& ext : extensions)
        {
            if (std::string_view(ext.extensionName) == VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
            {
                features.dynamicRendering = true;
            }
            if (std::string_view(ext.extensionName) == VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
            {
                features.synchronization2 = true;
            }
            if (std::string_view(ext.extensionName) == VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)
            {
                features.timelineSemaphore = true;
            }
        }
    }

    features.maxMsaaSamples = GetMaxUsableSampleCount(device);
    return features;
}

vk::SampleCountFlagBits mvk::util::GetMaxUsableSampleCount(vk::PhysicalDevice device)
{
    const auto props = device.getProperties();
    const auto counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
}

uint32_t mvk::util::CalculateDeviceScore(const GPUInfo& info, vk::SurfaceKHR surface)
{
    uint32_t score = 0;
    std::ostringstream breakdown;

    const auto& props = info.properties;
    const auto& features = info.features;
    const auto& supported = info.supportedFeatures;

    switch (props.deviceType)
    {
    case vk::PhysicalDeviceType::eDiscreteGpu:
        score += 10000;
        breakdown << "Discrete GPU: +10000, ";
        break;
    case vk::PhysicalDeviceType::eIntegratedGpu:
        score += 5000;
        breakdown << "Integrated GPU: +5000, ";
        break;
    case vk::PhysicalDeviceType::eVirtualGpu:
        score += 3000;
        breakdown << "Virtual GPU: +3000, ";
        break;
    case vk::PhysicalDeviceType::eCpu:
        score += 1000;
        breakdown << "CPU: +1000, ";
        break;
    default:
        score += 500;
        breakdown << "Other: +500, ";
        break;
    }

    uint64_t totalMemory = GetTotalDeviceMemory(info.memoryProperties);
    uint32_t memoryScore = static_cast<uint32_t>(totalMemory / (1024 * 1024 * 256));
    memoryScore = std::min(memoryScore, 2000u);
    score += memoryScore;
    breakdown << "Memory (" << (totalMemory / (1024 * 1024)) << "MB): +" << memoryScore << ", ";

    if (props.limits.maxComputeWorkGroupInvocations > 0)
    {
        uint32_t computeScore = std::min(props.limits.maxComputeWorkGroupInvocations / 32u, 1000u);
        score += computeScore;
        breakdown << "Compute Units: +" << computeScore << ", ";
    }

    uint32_t textureScore = std::min(props.limits.maxImageDimension2D / 1024u, 500u);
    score += textureScore;
    breakdown << "Max Texture Size: +" << textureScore << ", ";

    if (supported.dynamicRendering)
    {
        score += 500;
        breakdown << "Dynamic Rendering: +500, ";
    }

    if (supported.synchronization2)
    {
        score += 300;
        breakdown << "Synchronization2: +300, ";
    }

    if (supported.timelineSemaphore)
    {
        score += 200;
        breakdown << "Timeline Semaphore: +200, ";
    }

    if (supported.bufferDeviceAddress)
    {
        score += 200;
        breakdown << "Buffer Device Address: +200, ";
    }

    if (supported.descriptorIndexing)
    {
        score += 150;
        breakdown << "Descriptor Indexing: +150, ";
    }

    uint32_t msaaScore = static_cast<uint32_t>(supported.maxMsaaSamples) * 50;
    score += msaaScore;
    breakdown << "Max MSAA (" << static_cast<uint32_t>(supported.maxMsaaSamples) << "x): +" << msaaScore << ", ";

    if (features.geometryShader)
    {
        score += 100;
        breakdown << "Geometry Shader: +100, ";
    }

    if (features.tessellationShader)
    {
        score += 100;
        breakdown << "Tessellation: +100, ";
    }

    if (features.samplerAnisotropy)
    {
        score += 50;
        breakdown << "Anisotropic Filtering: +50, ";
    }

    if (features.sampleRateShading)
    {
        score += 50;
        breakdown << "Sample Rate Shading: +50, ";
    }

    try
    {
        auto queueFamilies = FindQueueFamilies(info.device, surface);

        if (queueFamilies.compute.has_value() &&
            queueFamilies.compute != queueFamilies.graphics)
        {
            score += 100;
            breakdown << "Dedicated Compute Queue: +100, ";
        }

        if (queueFamilies.transfer.has_value() &&
            queueFamilies.transfer != queueFamilies.graphics)
        {
            score += 50;
            breakdown << "Dedicated Transfer Queue: +50, ";
        }
    }
    catch (...)
    {
    }

    switch (props.vendorID)
    {
    case 0x10DE: // NVIDIA
        score += 100;
        breakdown << "NVIDIA: +100, ";
        break;
    case 0x1002: // AMD
        score += 80;
        breakdown << "AMD: +80, ";
        break;
    case 0x8086: // Intel
        score += 60;
        breakdown << "Intel: +60, ";
        break;
    default:
        breakdown << "Other Vendor: +0, ";
        break;
    }

    uint32_t apiMajor = VK_VERSION_MAJOR(props.apiVersion);
    uint32_t apiMinor = VK_VERSION_MINOR(props.apiVersion);
    uint32_t versionScore = (apiMajor - 1) * 200 + apiMinor * 50;
    score += versionScore;
    breakdown << "Vulkan " << apiMajor << "." << apiMinor << ": +" << versionScore;

    const_cast<GPUInfo&>(info).scoreBreakdown = breakdown.str();
    return score;
}

uint64_t mvk::util::GetTotalDeviceMemory(const vk::PhysicalDeviceMemoryProperties& memProps)
{
    uint64_t totalMemory = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
    {
        if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
        {
            totalMemory = std::max(totalMemory, memProps.memoryHeaps[i].size);
        }
    }
    return totalMemory;
}

std::string mvk::util::GetDeviceTypeString(vk::PhysicalDeviceType type)
{
    switch (type)
    {
    case vk::PhysicalDeviceType::eDiscreteGpu: return "Discrete GPU";
    case vk::PhysicalDeviceType::eIntegratedGpu: return "Integrated GPU";
    case vk::PhysicalDeviceType::eVirtualGpu: return "Virtual GPU";
    case vk::PhysicalDeviceType::eCpu: return "CPU";
    case vk::PhysicalDeviceType::eOther: return "Other";
    default: return "Unknown";
    }
}

bool mvk::util::IsVulkan13OrLater(vk::PhysicalDevice device)
{
    const auto props = device.getProperties();
    return props.apiVersion >= VK_API_VERSION_1_3;
}

bool mvk::util::IsVulkan12OrLater(vk::PhysicalDevice device)
{
    const auto props = device.getProperties();
    return props.apiVersion >= VK_API_VERSION_1_2;
}

bool mvk::util::IsVulkan11OrLater(vk::PhysicalDevice device)
{
    const auto props = device.getProperties();
    return props.apiVersion >= VK_API_VERSION_1_1;
}

uint32_t mvk::util::GetVulkanVersion(vk::PhysicalDevice device)
{
    const auto props = device.getProperties();
    return props.apiVersion;
}

std::string mvk::util::GetVulkanVersionString(vk::PhysicalDevice device)
{
    if (device)
    {
        const auto props = device.getProperties();
        const uint32_t major = VK_VERSION_MAJOR(props.apiVersion);
        const uint32_t minor = VK_VERSION_MINOR(props.apiVersion);
        const uint32_t patch = VK_VERSION_PATCH(props.apiVersion);
        return std::format("{}.{}.{}", major, minor, patch);
    }

    uint32_t instanceVersion = 0;
    vkEnumerateInstanceVersion(&instanceVersion);
    return std::format("{}.{}.{}",
        VK_VERSION_MAJOR(instanceVersion),
        VK_VERSION_MINOR(instanceVersion),
        VK_VERSION_PATCH(instanceVersion));
}

std::tuple<uint32_t, uint32_t, uint32_t> mvk::util::GetVulkanVersionNumbers(vk::PhysicalDevice device)
{
    const auto props = device.getProperties();
    return {
        VK_VERSION_MAJOR(props.apiVersion),
        VK_VERSION_MINOR(props.apiVersion),
        VK_VERSION_PATCH(props.apiVersion)
    };
}

void mvk::util::LogSystemInfo()
{
    try
    {
        std::ostringstream oss;
        oss << "\n=== Vulkan System Information ===\n";

        uint32_t instanceVersion = 0;
        vkEnumerateInstanceVersion(&instanceVersion);
        oss << "Vulkan Instance Version: "
            << VK_VERSION_MAJOR(instanceVersion) << "."
            << VK_VERSION_MINOR(instanceVersion) << "."
            << VK_VERSION_PATCH(instanceVersion) << "\n";

        const auto layers = vk::enumerateInstanceLayerProperties();
        oss << "Available Validation Layers (" << layers.size() << "):" << "\n";
        for (const auto& layer : layers)
        {
            oss << "  " << layer.layerName << " (v" << layer.implementationVersion << ")" << "\n";
        }

        const auto extensions = vk::enumerateInstanceExtensionProperties();
        oss << " Instance Extensions (" << extensions.size() << "):" << "\n";
        for (const auto& ext : extensions)
        {
            oss << "  " << ext.extensionName << "\n";
        }

        oss << "=================================\n";
        mcore::Logger::Info(oss.str());
        mcore::Logger::Print();
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error logging system info: ", e.what());
        mcore::Logger::Print();
    }
}

void mvk::util::LogDeviceInfo(vk::PhysicalDevice device, const DeviceFeatures& features, const QueueFamilyIndices& queueFamilyIndices)
{
    const auto props = device.getProperties();

    std::ostringstream oss;
    oss << "\n=== Device Information ===\n";
    oss << "Selected GPU: "<< props.deviceName.data()<<"\n";
    oss << "Vulkan Version: "<< GetVulkanVersionString(device)<<"\n";
    oss << "Driver Version: "<< props.driverVersion<<"\n";

    oss << "Features:"<<"\n";
    oss << "  Dynamic Rendering: "<< (features.dynamicRendering ? "Yes" : "No")<<"\n";
    oss << "  Synchronization2: "<< (features.synchronization2 ? "Yes" : "No")<<"\n";
    oss << "  Timeline Semaphore: "<< (features.timelineSemaphore ? "Yes" : "No")<<"\n";
    oss << "  Max MSAA Samples: "<< static_cast<uint32_t>(features.maxMsaaSamples)<<"\n";

    oss << "Queue Families:"<<"\n";
    oss << "  Graphics: "<< queueFamilyIndices.GetGraphics()<<"\n";
    oss << "  Present: "<< queueFamilyIndices.GetPresent()<<"\n";
    oss << "  Compute: "<< queueFamilyIndices.GetCompute()<<"\n";
    oss << "  Transfer: "<< queueFamilyIndices.GetTransfer()<<"\n";
    oss << "=============================\n";

    mcore::Logger::Info(oss.str());
    mcore::Logger::Print();
}

void mvk::util::LogGPUSelectionInfo(const std::vector<GPUInfo>& gpuInfos, const GPUInfo& selected)
{
    std::ostringstream oss;
    oss << "\n=== GPU Selection Results ===\n";

    for (size_t i = 0; i < gpuInfos.size(); ++i)
    {
        const auto& gpu = gpuInfos[i];
        const bool isSelected = (gpu.device == selected.device);

        oss << (isSelected ? ">>> " : "    ") <<
            "[" << i + 1 << "] " <<
            gpu.properties.deviceName <<
            " (" << GetDeviceTypeString(gpu.properties.deviceType) << ")" << "\n";

        std::string scoreInfo = (isSelected ? ">>> " : "    ");
        scoreInfo += "    Score: " + std::to_string(gpu.score);

        if (isSelected)
            scoreInfo += " * SELECTED";
        oss << scoreInfo << "\n";

        oss << (isSelected ? ">>> " : "    ") <<
            "    Memory: " << (GetTotalDeviceMemory(gpu.memoryProperties) / (1024 * 1024)) << " MB" <<
            "<< Vulkan: " << VK_VERSION_MAJOR(gpu.properties.apiVersion) <<
            "." << VK_VERSION_MINOR(gpu.properties.apiVersion) <<
            "." << VK_VERSION_PATCH(gpu.properties.apiVersion) << "\n";

        if (isSelected)
        {
            oss << ">>> Score Breakdown: " << gpu.scoreBreakdown << "\n";
        }
    }
    oss << "=============================\n";

    mcore::Logger::Info(oss.str());
    mcore::Logger::Print();
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL mvk::util::DebugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    const char* severityStr = "";
    switch (messageSeverity)
    {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        severityStr = "[VERBOSE]";
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
        severityStr = "[INFO]";
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
        severityStr = "[WARNING]";
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
        severityStr = "[ERROR]";
        break;
    }

    const char* typeStr = "";
    if (messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral)
        typeStr = "[GENERAL]";
    else if (messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
        typeStr = "[VALIDATION]";
    else if (messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        typeStr = "[PERFORMANCE]";

    std::ostringstream oss;
    oss << "Vulkan " << severityStr << " " << typeStr << ": "
        << pCallbackData->pMessage << "\n";
    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
    {
        if (pCallbackData->objectCount > 0)
        {
            oss << "  Related objects:\n";
            for (uint32_t i = 0; i < pCallbackData->objectCount; ++i)
            {
                const auto& obj = pCallbackData->pObjects[i];
                oss << "    [" << i << "] "
                    << vk::to_string(obj.objectType)
                    << " (handle: 0x" << std::hex << obj.objectHandle << std::dec << ")\n";
                if (obj.pObjectName)
                {
                    oss << " name: " << obj.pObjectName << "\n";
                }
            }
        }
        mcore::Logger::Error(oss.str());
    }
    else
    {
        mcore::Logger::Debug(oss.str());
    }

    mcore::Logger::Print();
    return VK_FALSE;
}