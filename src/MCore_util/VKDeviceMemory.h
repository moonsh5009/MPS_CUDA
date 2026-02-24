#pragma once

#include "IDeviceMemory.h"

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mcuda
{
    class __MY_EXT_CLASS__ VKDeviceMemory : public IDeviceMemory
    {
    public:
        VKDeviceMemory() = default;
        ~VKDeviceMemory() override;
        VKDeviceMemory(const VKDeviceMemory&) = delete;
        VKDeviceMemory(VKDeviceMemory&& src) noexcept;
        VKDeviceMemory& operator=(const VKDeviceMemory&) = delete;
        VKDeviceMemory& operator=(VKDeviceMemory&& src) noexcept;

        virtual void Initialize(vk::BufferUsageFlags usage);
        void Create(size_t byteSize) override;
        void Destroy() override;

        constexpr const vk::Buffer& GetBuffer() const { return m_buffer; }
		operator const vk::Buffer& () const { return m_buffer; }

    protected:
        void CreateVulkanBuffer(size_t byteSize);
        void ImportFromVulkan();

        vk::BufferUsageFlags m_usage;
        vk::DeviceMemory m_memory;
        vk::Buffer m_buffer;

    #ifdef _WIN32
        HANDLE m_memoryHandle = nullptr;
    #else
        int m_memoryHandle = -1;
    #endif
        void* m_cudaExtMem = nullptr;
    };
}

#include "HeaderPost.h"