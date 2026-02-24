#pragma once

#include "EnumExt.h"

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
    class SimpleCommandBuffer
    {
    public:
		SimpleCommandBuffer() = delete;
        SimpleCommandBuffer(vk::CommandBuffer cmdBuffer, vk::CommandPool cmdPool, mvk::QueueType queueType)
            : commandBuffer{ cmdBuffer }, commandPool{ cmdPool }, queueType{ queueType }
		{}
		SimpleCommandBuffer(const SimpleCommandBuffer&) = delete;
        SimpleCommandBuffer(SimpleCommandBuffer&& other) noexcept
        {
            commandBuffer = other.commandBuffer;
            commandPool = other.commandPool;
            other.commandBuffer = nullptr;
			other.commandPool = nullptr;
        }
		SimpleCommandBuffer& operator=(const SimpleCommandBuffer&) = delete;
        SimpleCommandBuffer& operator=(SimpleCommandBuffer&& other) noexcept
        {
            if (this != &other)
            {
                commandBuffer = other.commandBuffer;
                commandPool = other.commandPool;
				other.commandBuffer = nullptr;
                other.commandPool = nullptr;
            }
            return *this;
		}

        operator vk::CommandBuffer& () { return commandBuffer; }
        operator const vk::CommandBuffer&() const { return commandBuffer; }

		std::tuple<vk::CommandBuffer, vk::CommandPool, mvk::QueueType> Get()&&
        {
			return { commandBuffer, commandPool, queueType };
        }

    private:
        vk::CommandBuffer commandBuffer;
        vk::CommandPool commandPool;
		mvk::QueueType queueType;
    };

    class Commander;
    class __MY_EXT_CLASS__ VulkanCore
    {
    public:
        VulkanCore();
        ~VulkanCore();
        VulkanCore(const VulkanCore&) = delete;
        VulkanCore(VulkanCore&&) = default;
        VulkanCore& operator=(const VulkanCore&) = delete;
        VulkanCore& operator=(VulkanCore&&) = default;

		static VulkanCore* Instance();
        static void Initialize(HWND window);
        static void ShutDown();

        void RequestDevice(vk::SurfaceKHR surface);
		void WaitIdle() const;

        vk::SurfaceKHR CreateSurface(HWND window) const;
        uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

        SimpleCommandBuffer CreateSimpleCommandBuffer(mvk::QueueType type) const;
        void SimpleSubmit(SimpleCommandBuffer&& simpleCommandBuffer) const;

    #ifdef _WIN32
        HANDLE GetVulkanMemoryHandle(vk::DeviceMemory memory) const;
    #else
        int GetVulkanMemoryHandle(vk::DeviceMemory memory) const;
    #endif

        vk::Instance GetInstance() const { return m_instance; }
        vk::PhysicalDevice GetPhysicalDevice() const { return m_physical; }
        vk::Device GetDevice() const { return m_logical; }
        vk::Queue GetPresentQueue() const { return m_presentQueue; }
        vk::Queue GetGraphicQueue() const { return m_queues[mcore::to_sizet(QueueType::GRAPHIC)]; }
        vk::Queue GetComputeQueue() const { return m_queues[mcore::to_sizet(QueueType::COMPUTE)]; }
        vk::Queue GetTransferQueue() const { return m_queues[mcore::to_sizet(QueueType::TRANSFER)]; }
        vk::Queue GetQueue(QueueType type = QueueType::GRAPHIC) const { return m_queues[mcore::to_sizet(type)]; }

        const DeviceFeatures& GetFeatures() const { return m_features; }
        const QueueFamilyIndices& GetQueueFamilyIndices() const { return m_queueFamilyIndices; }

        bool IsOneQueueFamily() const { return m_queueFamilyIndices.graphics == m_queueFamilyIndices.present; }
        bool SupportsDynamicRendering() const { return m_features.dynamicRendering; }
        bool SupportsSynchronization2() const { return m_features.synchronization2; }
        bool SupportsTimelineSemaphore() const { return m_features.timelineSemaphore; }
        vk::SampleCountFlagBits GetMaxMsaaSamples() const { return m_features.maxMsaaSamples; }

    private:
        void GenerateInstance();
        void SelectPhysicalDevice(vk::SurfaceKHR surface);
        void GenerateLogicalDevice(vk::SurfaceKHR surface);

        bool IsVulkan13OrLater();

        bool LoadAllExtensionFunctions();

    private:
        vk::Instance m_instance;
        vk::PhysicalDevice m_physical;
        vk::Device m_logical;
        vk::Queue m_presentQueue;
        std::array<vk::Queue, mcore::enum_size<QueueType>()> m_queues;

        mvk::DeviceFeatures m_features;
        mvk::QueueFamilyIndices m_queueFamilyIndices;
        vk::DebugUtilsMessengerEXT m_debugMessenger;

    #ifdef _WIN32
        PFN_vkGetMemoryWin32HandleKHR m_pFnGetMemoryWin32HandleKHR = nullptr;
    #else
        PFN_vkGetMemoryFdKHR m_pFnGetMemoryFdKHR = nullptr;
    #endif
    };
}

#include "HeaderPost.h"
