#pragma once

#include "PersistentMappedMemory.h"

namespace mvk
{
    template<typename DATA, bool CUDA_EXPORT = false>
    class PersistentMappedBuffer : public PersistentMappedMemory
    {
        static_assert(std::is_trivially_copyable_v<DATA>, "DATA must be trivially copyable for GPU memory operations");
        static_assert(!std::is_pointer_v<DATA>, "DATA cannot be a pointer type");

    public:
        PersistentMappedBuffer(vk::BufferUsageFlags usage)
            : PersistentMappedMemory{ usage }
        {
            if constexpr (CUDA_EXPORT == false)
            {
                Create(sizeof(DATA));
            }
            else
            {
                CreateWithExport(sizeof(DATA));
            }
        }

        void CopyFromHost(const DATA& src);
        void CopyFromHost(const DATA& src, size_t offset);
        void CopyFromHost(const DATA& src, size_t offset, size_t byteLength);

        void CopyToHost(DATA& dst) const;
        void CopyToHost(DATA& dst, size_t offset) const;
        void CopyToHost(DATA& dst, size_t offset, size_t byteLength) const;

        void CopyFromHostAndUpload(vk::CommandBuffer cmdBuffer, const DATA& src);
        void CopyFromHostAndUpload(vk::CommandBuffer cmdBuffer, const DATA& src, size_t offset);
        void CopyFromHostAndUpload(vk::CommandBuffer cmdBuffer, const DATA& src, size_t offset, size_t byteLength);

        void DownloadAndCopyToHost(vk::CommandBuffer cmdBuffer, DATA& dst);
        void DownloadAndCopyToHost(vk::CommandBuffer cmdBuffer, DATA& dst, size_t offset);
        void DownloadAndCopyToHost(vk::CommandBuffer cmdBuffer, DATA& dst, size_t offset, size_t byteLength);

        void CopyFromDevice(vk::CommandBuffer cmdBuffer, const VulkanMemory& src, const vk::BufferCopy& copyRegion);
        void CopyToDevice(vk::CommandBuffer cmdBuffer, VulkanMemory& dst, const vk::BufferCopy& copyRegion) const;

        void Download(vk::CommandBuffer cmdBuffer, const vk::BufferCopy& region) { PersistentMappedMemory::Download(cmdBuffer, region); }
        void Upload(vk::CommandBuffer cmdBuffer) { PersistentMappedMemory::Upload(cmdBuffer); }

        void Download(vk::CommandBuffer cmdBuffer)
        {
            Download(cmdBuffer, { 0, 0, sizeof(DATA) });
        }

    private:
        bool ValidateRange(size_t offset, size_t size) const
        {
            if (offset + size > sizeof(DATA))
            {
                mcore::Logger::Error("Range validation failed. Offset: ", offset,
                    ", Size: ", size, ", Element size: ", sizeof(DATA));
                return false;
            }
            return true;
        }
    };

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromHost(const DATA& src)
    {
        PersistentMappedMemory::CopyFromHost(&src, { 0, 0, sizeof(DATA) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromHost(const DATA& src, size_t offset)
    {
        if (!ValidateRange(offset, sizeof(DATA) - offset)) return;

        PersistentMappedMemory::CopyFromHost(&src, { offset, offset, sizeof(DATA) - offset });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromHost(const DATA& src, size_t offset, size_t byteLength)
    {
        if (!ValidateRange(offset, byteLength)) return;

        PersistentMappedMemory::CopyFromHost(&src, { offset, offset, byteLength });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyToHost(DATA& dst) const
    {
        PersistentMappedMemory::CopyToHost(&dst, { 0, 0, sizeof(DATA) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyToHost(DATA& dst, size_t offset) const
    {
        if (!ValidateRange(offset, sizeof(DATA) - offset)) return;

        PersistentMappedMemory::CopyToHost(&dst, { offset, offset, sizeof(DATA) - offset });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyToHost(DATA& dst, size_t offset, size_t byteLength) const
    {
        if (!ValidateRange(offset, byteLength)) return;

        PersistentMappedMemory::CopyToHost(&dst, { offset, offset, byteLength });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromHostAndUpload(vk::CommandBuffer cmdBuffer, const DATA& src)
    {
        CopyFromHost(src);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromHostAndUpload(vk::CommandBuffer cmdBuffer, const DATA& src, size_t offset)
    {
        CopyFromHost(src, offset);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromHostAndUpload(vk::CommandBuffer cmdBuffer, const DATA& src, size_t offset, size_t byteLength)
    {
        CopyFromHost(src, offset, byteLength);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::DownloadAndCopyToHost(vk::CommandBuffer cmdBuffer, DATA& dst)
    {
        Download(cmdBuffer, { 0, 0, sizeof(DATA) });
        CopyToHost(dst);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::DownloadAndCopyToHost(vk::CommandBuffer cmdBuffer, DATA& dst, size_t offset)
    {
        Download(cmdBuffer, { offset, offset, sizeof(DATA) - offset });
        CopyToHost(dst, offset);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::DownloadAndCopyToHost(vk::CommandBuffer cmdBuffer, DATA& dst, size_t offset, size_t byteLength)
    {
        Download(cmdBuffer, { offset, offset, byteLength });
        CopyToHost(dst, offset, byteLength);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyFromDevice(vk::CommandBuffer cmdBuffer, const VulkanMemory& src, const vk::BufferCopy& copyRegion)
    {
        PersistentMappedMemory::CopyFromDevice(cmdBuffer, src, copyRegion);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void PersistentMappedBuffer<DATA, CUDA_EXPORT>::CopyToDevice(vk::CommandBuffer cmdBuffer, VulkanMemory& dst, const vk::BufferCopy& copyRegion) const
    {
        PersistentMappedMemory::CopyToDevice(cmdBuffer, dst, copyRegion);
    }
}