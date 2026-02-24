#pragma once

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ VulkanMemory
    {
    public:
        VulkanMemory();
        VulkanMemory(vk::BufferUsageFlags usage);
        virtual ~VulkanMemory() = default;
        VulkanMemory(const VulkanMemory&) = delete;
        VulkanMemory(VulkanMemory&& other) noexcept;
        VulkanMemory& operator=(const VulkanMemory&) = delete;
        VulkanMemory& operator=(VulkanMemory&& other)  noexcept;

		void Initialize(vk::BufferUsageFlags usage);

        constexpr vk::DeviceSize GetByteLength() const { return m_byteSize; }
        constexpr const vk::Buffer& GetBuffer() const { return m_buffer; }

		operator const vk::Buffer&() const { return m_buffer; }

    protected:
        virtual void Create(vk::DeviceSize byteSize) = 0;
        virtual void CreateWithExport(vk::DeviceSize byteSize) = 0;
        virtual void Destroy() = 0;

        virtual void CopyFromHost(const void* src, const vk::BufferCopy& copyRegion) = 0;
        virtual void CopyToHost(void* dst, const vk::BufferCopy& copyRegion) const = 0;

        virtual void Download(vk::CommandBuffer cmdBuffer, const vk::BufferCopy& region) = 0;
        virtual void Upload(vk::CommandBuffer cmdBuffer) = 0;

        void CopyFromDevice(vk::CommandBuffer cmdBuffer, const VulkanMemory& src, const vk::BufferCopy& copyRegion);
        void CopyToDevice(vk::CommandBuffer cmdBuffer, VulkanMemory& dst, const vk::BufferCopy& copyRegion) const;

        constexpr bool IsDirty() const { return std::get<0>(m_dirtyRange) < std::get<1>(m_dirtyRange); }

        vk::Buffer m_buffer;
        vk::DeviceMemory m_memory;
        vk::DeviceSize m_byteSize;
        vk::BufferUsageFlags m_usage;
        std::pair<vk::DeviceSize, vk::DeviceSize> m_dirtyRange;
    };
}

#include "HeaderPost.h"