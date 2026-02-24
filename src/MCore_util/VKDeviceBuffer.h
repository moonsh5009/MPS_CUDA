#pragma once

#include "VKDeviceMemory.h"
#include "IDeviceBuffer.h"

namespace mcuda
{
    template<typename T>
    class VKDeviceBuffer : public VKDeviceMemory, public IDeviceBuffer<T>
    {
    public:
        using value_type = T;
        using size_type = size_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;

		VKDeviceBuffer() = default;
        VKDeviceBuffer(vk::BufferUsageFlags usage);
        ~VKDeviceBuffer() override = default;
        VKDeviceBuffer(const VKDeviceBuffer&) = delete;
        VKDeviceBuffer(VKDeviceBuffer&& src) noexcept = default;
        VKDeviceBuffer& operator=(const VKDeviceBuffer&) = delete;
        VKDeviceBuffer& operator=(VKDeviceBuffer&& src) noexcept = default;

        void Initialize(vk::BufferUsageFlags usage) override;

        void Clear() override;
        void SetSize(size_t size) override;
        void Resize(size_t size) override;
        void Resize(size_t size, const T& value) override;
        void Reserve(size_t size) override;
        void ShrinkToFit() override;

        void CopyToDevice(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const override;
        void CopyFromDevice(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const override;
        void CopyToHost(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const override;
        void CopyFromHost(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const override;

    #ifdef MCORE_USE_CUDA
        void CopyToDeviceAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const override;
        void CopyFromDeviceAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const override;
        void CopyToHostAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const override;
        void CopyFromHostAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const override;
	#endif

        T* GetData() const override { return reinterpret_cast<T*>(IDeviceMemory::GetRawPointer()); }
    };
    template<typename T>
    VKDeviceBuffer<T>::VKDeviceBuffer(vk::BufferUsageFlags usage)
    {
        VKDeviceBuffer::Initialize(usage);
    }

    template<typename T>
    void VKDeviceBuffer<T>::Initialize(vk::BufferUsageFlags usage)
    {
        VKDeviceMemory::Initialize(usage | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    }

    template<typename T>
    void VKDeviceBuffer<T>::Clear()
    {
        SetSize(0);
    }

    template<typename T>
    void VKDeviceBuffer<T>::SetSize(size_t size)
    {
        const auto newCapacity = IDeviceBuffer<T>::NewCapacity(size);
        if (newCapacity != this->m_capacity)
        {
            this->m_capacity = newCapacity;
            this->Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
        }
        this->m_size = size;
    }

    template<typename T>
    void VKDeviceBuffer<T>::Resize(size_t size)
    {
        Reserve(size);
        this->m_size = size;
    }

    template<typename T>
    void VKDeviceBuffer<T>::Resize(size_t size, const T& value)
    {
        const auto oldSize = this->m_size;
        Resize(size);

        if (size > oldSize)
        {
            IDeviceBuffer<T>::Fill(value, size - oldSize, oldSize);
        }
    }

    template<typename T>
    void VKDeviceBuffer<T>::Reserve(size_t size)
    {
        const auto newCapacity = IDeviceBuffer<T>::NewCapacity(size);
        if (newCapacity != this->m_capacity)
        {
            this->m_capacity = size;
            if (this->m_size == 0)
            {
                this->Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
            }
            else
            {
                VKDeviceMemory tempMemory;
                tempMemory.Initialize(this->m_usage);
                tempMemory.Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
                tempMemory._CopyFromDevice(*this, IDeviceBuffer<T>::GetElementOffset(this->m_size), 0, 0);
                *static_cast<VKDeviceMemory*>(this) = std::move(tempMemory);
            }
        }
    }

    template<typename T>
    void VKDeviceBuffer<T>::ShrinkToFit()
    {
        if (this->m_capacity > this->m_size)
        {
            this->m_capacity = this->m_size;

            if (this->m_size == 0)
            {
                Destroy();
            }
            else
            {
                VKDeviceMemory tempMemory;
                tempMemory.Initialize(this->m_usage);
                tempMemory.Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
                tempMemory._CopyFromDevice(*this, IDeviceBuffer<T>::GetElementOffset(this->m_size), 0, 0);
                *static_cast<VKDeviceMemory*>(this) = std::move(tempMemory);
            }
        }
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyToDevice(const T* dst, uint64_t byteLength, uint64_t offset, uint64_t dstOffset) const
    {
        _CopyToDevice(const_cast<T*>(dst), byteLength, offset, dstOffset);
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyFromDevice(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset) const
    {
        _CopyFromDevice(src, byteLength, offset, srcOffset);
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyToHost(const T* dst, uint64_t byteLength, uint64_t offset, uint64_t dstOffset) const
    {
        _CopyToHost(const_cast<T*>(dst), byteLength, offset, dstOffset);
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyFromHost(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset) const
    {
        _CopyFromHost(src, byteLength, offset, srcOffset);
    }

#ifdef MCORE_USE_CUDA
    template<typename T>
    void VKDeviceBuffer<T>::CopyToDeviceAsync(const T* dst, uint64_t byteLength, uint64_t offset, uint64_t dstOffset, cudaStream_t stream) const
    {
        _CopyToDeviceAsync(const_cast<T*>(dst), byteLength, offset, dstOffset, stream);
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyFromDeviceAsync(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset, cudaStream_t stream) const
    {
        _CopyFromDeviceAsync(src, byteLength, offset, srcOffset, stream);
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyToHostAsync(const T* dst, uint64_t byteLength, uint64_t offset, uint64_t dstOffset, cudaStream_t stream) const
    {
        _CopyToHostAsync(const_cast<T*>(dst), byteLength, offset, dstOffset, stream);
    }

    template<typename T>
    void VKDeviceBuffer<T>::CopyFromHostAsync(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset, cudaStream_t stream) const
    {
        _CopyFromHostAsync(src, byteLength, offset, srcOffset, stream);
    }
#endif
}