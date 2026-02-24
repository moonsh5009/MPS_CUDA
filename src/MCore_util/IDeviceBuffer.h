#pragma once

#include "MCudaUtil.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <span>

namespace mcuda
{
    template<typename T>
    class IDeviceBuffer
    {
    public:
        IDeviceBuffer() : m_size{ 0 }, m_capacity{ 0 } {}
        virtual ~IDeviceBuffer() = default;
        IDeviceBuffer(const IDeviceBuffer&) = delete;
        IDeviceBuffer(IDeviceBuffer&&) noexcept = default;
        IDeviceBuffer& operator=(const IDeviceBuffer&) = delete;
        IDeviceBuffer& operator=(IDeviceBuffer&&) noexcept = default;

        virtual void Clear() = 0;
        virtual void SetSize(size_t size) = 0;
        virtual void Resize(size_t size) = 0;
        virtual void Resize(size_t size, const T& value) = 0;
        virtual void Reserve(size_t size) = 0;
        virtual void ShrinkToFit() = 0;

        virtual void CopyToDevice(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const = 0;
        virtual void CopyFromDevice(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const = 0;
        virtual void CopyToHost(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const = 0;
        virtual void CopyFromHost(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const = 0;

        void CopyToDevice(const IDeviceBuffer<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const
        {
            CopyToDevice(src.GetData(), byteLength, offset, dstOffset);
        }
        void CopyToDevice(thrust::device_vector<T>& src, uint64_t byteLength, uint64_t offset, uint64_t dstOffset) const
        {
            CopyToDevice(thrust::raw_pointer_cast(src.data()), byteLength, offset, dstOffset);
        }

        void CopyFromDevice(const IDeviceBuffer<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const
        {
            CopyFromDevice(src.GetData(), byteLength, offset, srcOffset);
        }
        void CopyFromDevice(const thrust::device_vector<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const
        {
            CopyFromDevice(thrust::raw_pointer_cast(src.data()), byteLength, offset, srcOffset);
        }

        void CopyToHost(std::vector<T>& dst, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const
        {
            CopyToHost(dst.data(), byteLength, offset, dstOffset);
        }
        void CopyToHost(std::span<T> dst, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const
        {
            CopyToHost(dst.data(), byteLength, offset, dstOffset);
        }
        void CopyToHost(thrust::host_vector<T>& dst, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0) const
        {
            CopyToHost(dst.data(), byteLength, offset, dstOffset);
        }

        void CopyFromHost(const std::vector<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const
        {
            CopyFromHost(src.data(), byteLength, offset, srcOffset);
        }
        void CopyFromHost(const std::span<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const
        {
            CopyFromHost(src.data(), byteLength, offset, srcOffset);
        }
        void CopyFromHost(const thrust::host_vector<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0) const
        {
            CopyFromHost(src.data(), byteLength, offset, srcOffset);
        }

    #ifdef MCORE_USE_CUDA
        virtual void CopyToDeviceAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const = 0;
        virtual void CopyFromDeviceAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const = 0;
        virtual void CopyToHostAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const = 0;
        virtual void CopyFromHostAsync(const T* src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const = 0;

        void CopyToDeviceAsync(const IDeviceBuffer<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const
        {
            CopyToDeviceAsync(src.GetData(), byteLength, offset, dstOffset, stream);
        }
        void CopyToDeviceAsync(const thrust::device_vector<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const
        {
            CopyToDeviceAsync(thrust::raw_pointer_cast(src.data()), byteLength, offset, dstOffset, stream);
        }

        void CopyFromDeviceAsync(const IDeviceBuffer<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const
        {
            CopyFromDeviceAsync(src.GetData(), byteLength, offset, srcOffset, stream);
        }
        void CopyFromDeviceAsync(const thrust::device_vector<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const
        {
            CopyFromDeviceAsync(thrust::raw_pointer_cast(src.data()), byteLength, offset, srcOffset, stream);
        }

        void CopyToHostAsync(std::vector<T>& dst, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const
        {
            CopyToHostAsync(dst.data(), byteLength, offset, dstOffset, stream);
        }
        void CopyToHostAsync(std::span<T> dst, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const
        {
            CopyToHostAsync(dst.data(), byteLength, offset, dstOffset, stream);
        }
        void CopyToHostAsync(thrust::host_vector<T>& dst, uint64_t byteLength, uint64_t offset = 0, uint64_t dstOffset = 0, cudaStream_t stream = 0) const
        {
            CopyToHostAsync(dst.data(), byteLength, offset, dstOffset, stream);
        }

        void CopyFromHostAsync(const std::vector<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const
        {
            CopyFromHostAsync(src.data(), byteLength, offset, srcOffset, stream);
        }
        void CopyFromHostAsync(const std::span<T>& src, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const
        {
            CopyFromHostAsync(src.data(), byteLength, offset, srcOffset, stream);
        }
        void CopyFromHostAsync(const thrust::host_vector<T>& dst, uint64_t byteLength, uint64_t offset = 0, uint64_t srcOffset = 0, cudaStream_t stream = 0) const
        {
            CopyFromHostAsync(dst.data(), byteLength, offset, srcOffset, stream);
        }
	#endif

        void Fill(const T& value, size_t count, size_t offset = 0)
        {
            thrust::device_ptr<T> start = thrust::device_pointer_cast(GetData() + offset);

        #ifdef __CUDACC__
            thrust::fill(start, start + count, value);
        #endif
        }

        virtual T* GetData() const = 0;

        constexpr size_t GetSize() const { return m_size; }
        constexpr size_t GetByteLength() const { return m_size * sizeof(T); }
        constexpr size_t GetCapacity() const { return m_capacity; }
        constexpr size_t GetElementByteLength() const { return sizeof(T); }
        constexpr size_t GetElementOffset(size_t index) const { return sizeof(T) * index; }

		bool IsEmpty() const { return m_size == 0; }

        thrust::device_ptr<T> begin() const { return thrust::device_pointer_cast(GetData()); }
        thrust::device_ptr<T> end() const { return thrust::device_pointer_cast(GetData() + m_size); }

    protected:
        constexpr size_t NewCapacity(size_t newSize) const
        {
            const auto oldCapacity = GetCapacity();
            if (oldCapacity >= newSize)
                return oldCapacity;

            const auto geometric = oldCapacity + (oldCapacity >> 1);
            if (geometric < newSize)
                return newSize;
            return geometric;
        }

        size_t m_size;
        size_t m_capacity;
    };
}