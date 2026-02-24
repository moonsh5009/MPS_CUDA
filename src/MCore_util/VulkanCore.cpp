#include "stdafx.h"
#include "VulkanCore.h"

#include "VulkanUtil.h"
#include "Commander.h"

using namespace mcore;

static std::unique_ptr<mvk::VulkanCore> m_singleton;

mvk::VulkanCore::VulkanCore()
{
}

mvk::VulkanCore::~VulkanCore()
{
    try
    {
        m_logical.destroy();
        m_logical = nullptr;

        m_physical = nullptr;
        m_presentQueue = nullptr;
        for (auto& queue : m_queues)
        {
            queue = nullptr;
        }
        m_queueFamilyIndices.reset();

        if (m_instance)
        {
            util::DestroyDebugMessenger(m_instance, m_debugMessenger);
            m_debugMessenger = nullptr;

            m_instance.destroy();
            m_instance = nullptr;
        }

        mcore::Logger::Debug("Vulkan VulkanCore destroyed successfully");
        mcore::Logger::Print();
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error during VulkanCore destruction: ", e.what());
        mcore::Logger::Print();
    }
}

mvk::VulkanCore* mvk::VulkanCore::Instance()
{
#ifdef _DEBUG
    if (!m_singleton)
    {
		mcore::Logger::Warning("VulkanCore is not initialized yet!");
    }
#endif
    return m_singleton.get();
}

void mvk::VulkanCore::Initialize(HWND window)
{
    util::LogSystemInfo();

#ifdef _DEBUG
    if (m_singleton)
    {
        mcore::Logger::Warning("VulkanCore is already initialized!");
    }
#endif

	m_singleton = std::make_unique<mvk::VulkanCore>();
	m_singleton->GenerateInstance();

    auto surface = m_singleton->CreateSurface(window);
    if (!surface)
    {
        throw std::runtime_error("Can't Create Vulkan Win32 Surface");
    }

    m_singleton->RequestDevice(surface);
    m_singleton->GetInstance().destroySurfaceKHR(surface);
}

void mvk::VulkanCore::ShutDown()
{
    m_singleton.reset();
}

void mvk::VulkanCore::RequestDevice(vk::SurfaceKHR surface)
{
    SelectPhysicalDevice(surface);
    GenerateLogicalDevice(surface);

    LoadAllExtensionFunctions();

    util::LogDeviceInfo(m_physical, m_features, m_queueFamilyIndices);
}

void mvk::VulkanCore::WaitIdle() const
{
    GetDevice().waitIdle();
}

vk::SurfaceKHR mvk::VulkanCore::CreateSurface(HWND window) const
{
    vk::Win32SurfaceCreateInfoKHR createInfo({}, GetModuleHandle(nullptr), window);
    return m_instance.createWin32SurfaceKHR(createInfo);
}

uint32_t mvk::VulkanCore::FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
{
    const auto memProps = m_physical.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

mvk::SimpleCommandBuffer mvk::VulkanCore::CreateSimpleCommandBuffer(mvk::QueueType type) const
{
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eTransient;
    poolInfo.queueFamilyIndex = m_queueFamilyIndices.Get(type);
    auto commandPool = GetDevice().createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    auto commandBuffer = GetDevice().allocateCommandBuffers(allocInfo)[0];
    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	SimpleCommandBuffer simpleCmdBuffer{ commandBuffer, commandPool, type };
    commandBuffer.begin(beginInfo);
    return simpleCmdBuffer;
}

void mvk::VulkanCore::SimpleSubmit(SimpleCommandBuffer&& simpleCommandBuffer) const
{
    const auto [commandBuffer, commandPool, queueType] = std::move(simpleCommandBuffer).Get();
    commandBuffer.end();

    const auto& queue = GetQueue(queueType);

    vk::FenceCreateInfo fenceInfo;
    auto fence = GetDevice().createFence(fenceInfo);

    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    queue.submit(submitInfo, fence);

    GetDevice().waitForFences(fence, VK_TRUE, UINT64_MAX);

    GetDevice().destroyFence(fence);
    GetDevice().freeCommandBuffers(commandPool, commandBuffer);
    GetDevice().destroyCommandPool(commandPool);
}

#ifdef _WIN32
HANDLE mvk::VulkanCore::GetVulkanMemoryHandle(vk::DeviceMemory memory) const
{
    if (!m_pFnGetMemoryWin32HandleKHR)
    {
        printf("Extension function not loaded\n");
        return nullptr;
    }

    HANDLE outHandle;

    VkMemoryGetWin32HandleInfoKHR getHandleInfo = {};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.pNext = nullptr;
    getHandleInfo.memory = static_cast<VkDeviceMemory>(memory);
    getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkResult result = m_pFnGetMemoryWin32HandleKHR(
        static_cast<VkDevice>(m_logical),
        &getHandleInfo,
        &outHandle
    );

    if (result != VK_SUCCESS)
    {
        printf("vkGetMemoryWin32HandleKHR failed with error: %d\n", result);
        return nullptr;
    }

    if (outHandle == nullptr || outHandle == INVALID_HANDLE_VALUE)
    {
        printf("Invalid handle returned\n");
        return nullptr;
    }

    printf("Successfully got Win32 handle: 0x%p\n", outHandle);
    return outHandle;
}
#else
int mvk::VulkanCore::GetVulkanMemoryHandle(vk::DeviceMemory memory) const
{
    if (!m_pFnGetMemoryFdKHR)
    {
        printf("Extension function not loaded\n");
        return nullptr;
    }

    int outHandle;

    /*VkMemoryGetWin32HandleInfoKHR getHandleInfo = {};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.pNext = nullptr;
    getHandleInfo.memory = static_cast<VkDeviceMemory>(memory);
    getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkResult result = m_pFnGetMemoryFdKHR(
        static_cast<VkDevice>(m_logical),
        &getHandleInfo,
        &outHandle
    );

    if (result != VK_SUCCESS)
    {
        printf("vkGetMemoryWin32HandleKHR failed with error: %d\n", result);
        return nullptr;
    }

    if (outHandle == nullptr || outHandle == INVALID_HANDLE_VALUE)
    {
        printf("Invalid handle returned\n");
        return nullptr;
    }

    printf("Successfully got Win32 handle: 0x%p\n", outHandle);*/
    return outHandle;
}
#endif

void mvk::VulkanCore::GenerateInstance()
{
    try
    {
        if constexpr (ENABLE_VALIDATION_LAYERS)
        {
            if (!util::CheckValidationLayerSupport())
            {
                throw std::runtime_error("Validation layers requested, but not available!");
            }
        }

        std::string appName = "Vulkan Application";
        std::string engineName = "MVK Engine";
        vk::ApplicationInfo appInfo(
            appName.c_str(),
            VK_MAKE_VERSION(1, 0, 0),
            engineName.c_str(),
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_3
        );

        vk::InstanceCreateInfo createInfo({}, &appInfo);

        const auto extensions = util::GetRequiredExtensions();
        createInfo.setPEnabledExtensionNames(extensions);

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if constexpr (ENABLE_VALIDATION_LAYERS)
        {
            debugCreateInfo = util::PopulateDebugMessengerCreateInfo();
            createInfo.setPEnabledLayerNames(util::VALIDATION_LAYERS);
            createInfo.setPNext(&debugCreateInfo);
        }

        try
        {
            m_instance = vk::createInstance(createInfo);
        }
        catch (const vk::SystemError& e)
        {
            mcore::Logger::Error("Failed to create Vulkan instance: ", e.what());
            mcore::Logger::Print();
            throw std::runtime_error("Vulkan instance creation failed");
        }

        vk::DebugUtilsMessengerEXT debugMessenger;
        if constexpr (ENABLE_VALIDATION_LAYERS)
        {
            try
            {
                m_debugMessenger = util::SetupDebugMessenger(m_instance);
            }
            catch (const std::exception& e)
            {
                mcore::Logger::Error("Warning: Failed to setup debug messenger: ", e.what());
                mcore::Logger::Print();
            }
        }

        mcore::Logger::Debug("Vulkan instance created successfully");
        mcore::Logger::Print();
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan System Error: ", e.what());
        mcore::Logger::Error("Error Code: ", e.code());
        mcore::Logger::Print();
        throw std::runtime_error("Failed to initialize Vulkan VulkanCore");
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Initialization Error: ", e.what());
        mcore::Logger::Print();
        throw;
    }
}

void mvk::VulkanCore::SelectPhysicalDevice(vk::SurfaceKHR surface)
{
    const auto physicalDevices = m_instance.enumeratePhysicalDevices();
    if (physicalDevices.empty())
    {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    mcore::Logger::Info("Found ", physicalDevices.size(), " GPU(s)");

    std::vector<GPUInfo> gpuInfos;
    gpuInfos.reserve(physicalDevices.size());

    for (const auto& device : physicalDevices)
    {
        try
        {
            auto gpuInfo = util::EvaluatePhysicalDevice(device, surface);
            if (gpuInfo.score > 0)
            {
                gpuInfos.push_back(std::move(gpuInfo));
            }
        }
        catch (const std::exception& e)
        {
            mcore::Logger::Error("Error evaluating GPU: ", e.what());
            mcore::Logger::Print();
            continue;
        }
    }

    if (gpuInfos.empty())
    {
        throw std::runtime_error("No suitable GPU found!");
    }

    std::sort(gpuInfos.begin(), gpuInfos.end(), std::greater<GPUInfo>());

    const auto& bestGPU = gpuInfos[0];
    m_physical = bestGPU.device;

    util::LogGPUSelectionInfo(gpuInfos, bestGPU);

    mcore::Logger::Info("Selected GPU: ", bestGPU.properties.deviceName
        , " (Score: ", bestGPU.score, ")");
    mcore::Logger::Print();
}

void mvk::VulkanCore::GenerateLogicalDevice(vk::SurfaceKHR surface)
{
    auto queueFamilies = util::FindQueueFamilies(m_physical, surface);
    auto features = util::QueryDeviceFeatures(m_physical);

    const auto queuePriority = 1.0f;
    const auto queueFamilyIndices = queueFamilies.GetAllIndices();

    vk::PhysicalDeviceFeatures2 deviceFeatures2{};
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
    vk::PhysicalDeviceSynchronization2Features sync2Features{};
	vk::PhysicalDeviceRobustness2FeaturesEXT robustness2Features{};

    deviceFeatures2.features.shaderFloat64 = VK_TRUE;
    deviceFeatures2.features.multiDrawIndirect = VK_TRUE;

    robustness2Features.nullDescriptor = VK_TRUE;
    deviceFeatures2.pNext = &robustness2Features;

    if (features.dynamicRendering)
    {
        dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
        dynamicRenderingFeatures.pNext = deviceFeatures2.pNext;
        deviceFeatures2.pNext = &dynamicRenderingFeatures;
    }
    if (features.timelineSemaphore)
    {
        timelineFeatures.timelineSemaphore = VK_TRUE;
        timelineFeatures.pNext = deviceFeatures2.pNext;
        deviceFeatures2.pNext = &timelineFeatures;
    }
    if (features.synchronization2)
    {
        sync2Features.synchronization2 = VK_TRUE;
        sync2Features.pNext = deviceFeatures2.pNext;
        deviceFeatures2.pNext = &sync2Features;
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.reserve(queueFamilyIndices.size());
    for (auto queueFamilyIndex : queueFamilyIndices)
    {
        queueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo
            {
                {},
                queueFamilyIndex,
                1,
                &queuePriority
            });
    }

    std::vector<const char*> extensions(util::DEVICE_EXTENSIONS.begin(), util::DEVICE_EXTENSIONS.end());
    if (features.dynamicRendering && !IsVulkan13OrLater())
    {
        extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        extensions.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    }
    if (features.synchronization2 && !IsVulkan13OrLater())
    {
        extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo deviceCreateInfo{
        {},
        queueCreateInfos,
        {},
        extensions,
        {},
        &deviceFeatures2
    };

    if constexpr (ENABLE_VALIDATION_LAYERS)
        deviceCreateInfo.setPEnabledLayerNames(util::VALIDATION_LAYERS);

    m_logical = m_physical.createDevice(deviceCreateInfo);

    m_presentQueue = m_logical.getQueue(*queueFamilies.present, 0);
    m_queues[to_sizet(QueueType::GRAPHIC)] = m_logical.getQueue(*queueFamilies.graphics, 0);
    if (queueFamilies.compute.has_value())
        m_queues[to_sizet(QueueType::COMPUTE)] = m_logical.getQueue(*queueFamilies.compute, 0);
    else
        m_queues[to_sizet(QueueType::COMPUTE)] = m_queues[to_sizet(QueueType::GRAPHIC)];
    if (queueFamilies.transfer.has_value())
        m_queues[to_sizet(QueueType::TRANSFER)] = m_logical.getQueue(*queueFamilies.transfer, 0);
    else
        m_queues[to_sizet(QueueType::TRANSFER)] = m_queues[to_sizet(QueueType::GRAPHIC)];

    m_queueFamilyIndices = std::move(queueFamilies);
    m_features = features;
}

bool mvk::VulkanCore::IsVulkan13OrLater()
{
    return util::IsVulkan13OrLater(m_physical);
}

bool mvk::VulkanCore::LoadAllExtensionFunctions()
{
#ifdef _WIN32
    m_pFnGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)
        vkGetDeviceProcAddr(m_logical, "vkGetMemoryWin32HandleKHR");

    if (!m_pFnGetMemoryWin32HandleKHR)
    {
        mcore::Logger::Error("Failed to load Win32 handle function");
        mcore::Logger::Print();
        return false;
    }
#else
    m_pFnGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)
        vkGetDeviceProcAddr(m_logical, "vkGetMemoryFdKHR");

    if (!m_pFnGetMemoryFdKHR)
    {
        mcore::Logger::Error("Note: Linux FD function not available (expected on Windows)");
        mcore::Logger::Print();
    }
#endif

    return true;
}