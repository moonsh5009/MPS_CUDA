#pragma once

#include "VulkanMemory.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ StagedMemory : public VulkanMemory
    {
    public:
        StagedMemory(vk::BufferUsageFlags usage);
        ~StagedMemory() override;
        StagedMemory(const StagedMemory&) = delete;
		StagedMemory(StagedMemory&& other) noexcept;
		StagedMemory& operator=(const StagedMemory&) = delete;
		StagedMemory& operator=(StagedMemory&& other) noexcept;

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
    };
}

#include "HeaderPost.h"