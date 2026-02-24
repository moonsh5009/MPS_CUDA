#pragma once

#include "VulkanMemory.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ PersistentMappedMemory : public VulkanMemory
    {
    public:
        PersistentMappedMemory(vk::BufferUsageFlags usage);
        ~PersistentMappedMemory() override;
		PersistentMappedMemory(const PersistentMappedMemory&) = delete;
		PersistentMappedMemory(PersistentMappedMemory&& other) noexcept;
		PersistentMappedMemory& operator=(const PersistentMappedMemory&) = delete;
		PersistentMappedMemory& operator=(PersistentMappedMemory&& other)  noexcept;

    protected:
        void Create(vk::DeviceSize byteSize) override;
        void CreateWithExport(vk::DeviceSize byteSize) override;
        void Destroy() override;

        void CopyFromHost(const void* src, const vk::BufferCopy& copyRegion) override;
        void CopyToHost(void* dst, const vk::BufferCopy& copyRegion) const override;

        void Download(vk::CommandBuffer cmdBuffer, const vk::BufferCopy& region) override;
        void Upload(vk::CommandBuffer cmdBuffer) override;

        bool IsValid() const { return m_buffer && m_stagingBuffer; }

    private:
        void CreateStagingBuffer(vk::DeviceSize byteSize);

        vk::Buffer m_stagingBuffer;
        vk::DeviceMemory m_stagingMemory;
        void* m_mappedStaging;
    };
}

#include "HeaderPost.h"