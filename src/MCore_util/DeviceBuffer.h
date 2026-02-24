#pragma once

#include "DeviceMemory.h"
#include "IDeviceBuffer.h"

namespace mcuda
{
    template<typename T>
    class DeviceBuffer : public DeviceMemory, public IDeviceBuffer<T>
    {
    public:
        using value_type = T;
        using size_type = size_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;

        DeviceBuffer() = default;
        ~DeviceBuffer() override = default;
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer(DeviceBuffer&& src) noexcept = default;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(DeviceBuffer&& src) noexcept = default;

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
    void DeviceBuffer<T>::Clear()
    {
        SetSize(0);
    }

    template<typename T>
    void DeviceBuffer<T>::SetSize(size_t size)
    {
        const auto newCapacity = this->NewCapacity(size);
        if (newCapacity != this->m_capacity)
        {
            this->m_capacity = newCapacity;
            this->Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
        }
        this->m_size = size;
    }

    template<typename T>
    void DeviceBuffer<T>::Resize(size_t size)
    {
        Reserve(size);
        this->m_size = size;
    }

    template<typename T>
    void DeviceBuffer<T>::Resize(size_t size, const T& value)
    {
        const auto oldSize = this->m_size;
        Resize(size);

        if (size > oldSize)
        {
            IDeviceBuffer<T>::Fill(value, size - oldSize, oldSize);
        }
    }

    template<typename T>
    void DeviceBuffer<T>::Reserve(size_t size)
    {
        const auto newCapacity = this->NewCapacity(size);
        if (newCapacity != this->m_capacity)
        {
            this->m_capacity = size;
            if (this->m_size == 0)
            {
                this->Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
            }
            else
            {
                DeviceMemory tempMemory;
                tempMemory.Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
                tempMemory._CopyFromDevice(*this, this->m_size * sizeof(T), 0, 0);
                *static_cast<DeviceMemory*>(this) = std::move(tempMemory);
            }
        }
    }

    template<typename T>
    void DeviceBuffer<T>::ShrinkToFit()
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
                DeviceMemory tempMemory;
                tempMemory.Create(IDeviceBuffer<T>::GetElementOffset(this->m_capacity));
                tempMemory._CopyFromDevice(*this, IDeviceBuffer<T>::GetElementOffset(this->m_size), 0, 0);
                *static_cast<DeviceMemory*>(this) = std::move(tempMemory);
            }
        }
    }

    template<typename T>
    void DeviceBuffer<T>::CopyToDevice(const T* dst, uint64_t byteLength, uint64_t offset, uint64_t dstOffset) const
    {
        _CopyToDevice(const_cast<T*>(dst), byteLength, offset, dstOffset);
    }

    template<typename T>
    void DeviceBuffer<T>::CopyFromDevice(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset) const
    {
        _CopyFromDevice(src, byteLength, offset, srcOffset);
    }

    template<typename T>
    void DeviceBuffer<T>::CopyToHost(const T* src, uint64_t byteLength, uint64_t offset, uint64_t dstOffset) const
    {
        _CopyToHost(const_cast<T*>(src), byteLength, offset, dstOffset);
    }

    template<typename T>
    void DeviceBuffer<T>::CopyFromHost(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset) const
    {
        _CopyFromHost(src, byteLength, offset, srcOffset);
    }

#ifdef MCORE_USE_CUDA
    template<typename T>
    void DeviceBuffer<T>::CopyToDeviceAsync(const T* dst, uint64_t byteLength, uint64_t offset, uint64_t dstOffset, cudaStream_t stream) const
    {
        _CopyToDeviceAsync(const_cast<T*>(dst), byteLength, offset, dstOffset, stream);
    }

    template<typename T>
    void DeviceBuffer<T>::CopyFromDeviceAsync(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset, cudaStream_t stream) const
    {
        _CopyFromDeviceAsync(src, byteLength, offset, srcOffset, stream);
    }

    template<typename T>
    void DeviceBuffer<T>::CopyToHostAsync(const T* src, uint64_t byteLength, uint64_t offset, uint64_t dstOffset, cudaStream_t stream) const
    {
        _CopyToHostAsync(const_cast<T*>(src), byteLength, offset, dstOffset, stream);
    }

    template<typename T>
    void DeviceBuffer<T>::CopyFromHostAsync(const T* src, uint64_t byteLength, uint64_t offset, uint64_t srcOffset, cudaStream_t stream) const
    {
        _CopyFromHostAsync(src, byteLength, offset, srcOffset, stream);
    }
#endif
}