#pragma once

#include "MCudaUtil.h"

#include "HeaderPre.h"

namespace mcuda
{
    class __MY_EXT_CLASS__ IDeviceMemory
    {
    public:
        IDeviceMemory() = default;
        virtual ~IDeviceMemory() = default;
        IDeviceMemory(const IDeviceMemory&) = delete;
        IDeviceMemory(IDeviceMemory&& src) noexcept;
        IDeviceMemory& operator=(const IDeviceMemory&) = delete;
        IDeviceMemory& operator=(IDeviceMemory&& src) noexcept;

        virtual void Create(size_t byteSize) = 0;
        virtual void Destroy() = 0;

        void _CopyFromDevice(const void* src, size_t size, size_t offset, size_t srcOffset) const;
        void _CopyFromDevice(const IDeviceMemory& src, size_t size, size_t offset, size_t srcOffset) const;
        
        void _CopyToDevice(void* dst, size_t size, size_t offset, size_t dstOffset) const;
        void _CopyToDevice(const IDeviceMemory& dst, size_t size, size_t offset, size_t dstOffset) const;
        
        void _CopyFromHost(const void* src, size_t size, size_t offset, size_t srcOffset) const;
        void _CopyToHost(void* dst, size_t size, size_t offset, size_t dstOffset) const;

    #ifdef MCORE_USE_CUDA
        void _CopyFromDeviceAsync(const void* src, size_t size, size_t offset, size_t srcOffset, cudaStream_t stream) const;
        void _CopyFromDeviceAsync(const IDeviceMemory& src, size_t size, size_t offset, size_t srcOffset, cudaStream_t stream) const;
        
        void _CopyToDeviceAsync(void* dst, size_t size, size_t offset, size_t dstOffset, cudaStream_t stream) const;
        void _CopyToDeviceAsync(const IDeviceMemory& dst, size_t size, size_t offset, size_t dstOffset, cudaStream_t stream) const;
        
        void _CopyFromHostAsync(const void* src, size_t size, size_t offset, size_t srcOffset, cudaStream_t stream) const;
        void _CopyToHostAsync(void* dst, size_t size, size_t offset, size_t dstOffset, cudaStream_t stream) const;
    #endif

        constexpr size_t GetCapacityByteLength() const { return m_byteSize; }
        constexpr void* GetRawPointer() const { return m_rawPtr; }

    protected:
        size_t m_byteSize = 0;
        void* m_rawPtr = nullptr;
    };
}

#include "HeaderPost.h"