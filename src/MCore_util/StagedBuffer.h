#pragma once

#include "StagedMemory.h"

#include <vector>
#include <span>

namespace mvk
{
    template<typename DATA, bool CUDA_EXPORT = false>
    class StagedBuffer : public StagedMemory
    {
        static_assert(std::is_trivially_copyable_v<DATA>, "DATA must be trivially copyable for GPU memory operations");
        static_assert(!std::is_pointer_v<DATA>, "DATA cannot be a pointer type");

    public:
        using value_type = DATA;
        using size_type = size_t;
        using reference = DATA&;
        using const_reference = const DATA&;
        using pointer = DATA*;
        using const_pointer = const DATA*;

        StagedBuffer(vk::BufferUsageFlags usage, size_t initialCapacity = 0)
            : StagedMemory{ usage }
            , m_size{ 0 }
            , m_capacity{ initialCapacity }
        {
            if constexpr (CUDA_EXPORT == false)
            {
                Create(sizeof(DATA) * m_capacity);
            }
            else
            {
                CreateWithExport(sizeof(DATA) * m_capacity);
            }
        }

        StagedBuffer(vk::BufferUsageFlags usage, const std::vector<DATA>& initialData)
            : StagedBuffer(usage, initialData.size())
        {
            Assign(initialData);
        }

        void SetSize(size_t size);
        void Resize(size_t size);
        void Resize(size_t size, const DATA& value);
        void Reserve(size_t capacity);
        void ShrinkToFit();
        void Clear() { m_size = 0; }

        void PushBack(const DATA& value);
        void PushBackAndUpload(vk::CommandBuffer cmdBuffer, const DATA& value);
        void PopBack();

        void Insert(size_t index, const DATA& value);
        void Insert(size_t index, const std::vector<DATA>& values);
        void Insert(size_t index, const DATA* values, size_t count);

        void Erase(size_t index);
        void Erase(size_t startIndex, size_t count);

        void Assign(const std::vector<DATA>& src);
        void Assign(const DATA* src, size_t count);
        void Assign(std::span<const DATA> src);
        void Assign(std::initializer_list<DATA> src);

        void SetElement(size_t index, const DATA& value);
        void SetElements(size_t startIndex, const std::vector<DATA>& values);
        void SetElements(size_t startIndex, const DATA* values, size_t count);
        void SetElements(size_t startIndex, std::span<const DATA> values);

        void SetRange(size_t startIndex, size_t count, const DATA& value);

        void GetElement(size_t index, DATA& dst) const;
        void GetElements(size_t startIndex, size_t count, std::vector<DATA>& dst) const;
        void GetElements(size_t startIndex, size_t count, DATA* dst) const;
        std::vector<DATA> ToVector() const;

        void AssignAndUpload(vk::CommandBuffer cmdBuffer, const std::vector<DATA>& src);
        void AssignAndUpload(vk::CommandBuffer cmdBuffer, const DATA* src, size_t count);
        void AssignAndUpload(vk::CommandBuffer cmdBuffer, std::span<const DATA> src);
        void AssignAndUpload(vk::CommandBuffer cmdBuffer, std::initializer_list<DATA> src);

        void SetElementAndUpload(vk::CommandBuffer cmdBuffer, size_t index, const DATA& value);
        void SetElementsAndUpload(vk::CommandBuffer cmdBuffer, size_t startIndex, const std::vector<DATA>& values);

        void DownloadAndGetElement(vk::CommandBuffer cmdBuffer, size_t index, DATA& dst);
        void DownloadAndGetElements(vk::CommandBuffer cmdBuffer, size_t startIndex, size_t count, std::vector<DATA>& dst);
        std::vector<DATA> DownloadToVector(vk::CommandBuffer cmdBuffer);

        void CopyFromArray(vk::CommandBuffer cmdBuffer, const StagedBuffer<DATA, CUDA_EXPORT>& src,
            size_t srcIndex = 0, size_t dstIndex = 0, size_t count = SIZE_MAX);
        void CopyToArray(vk::CommandBuffer cmdBuffer, StagedBuffer<DATA, CUDA_EXPORT>& dst,
            size_t srcIndex = 0, size_t dstIndex = 0, size_t count = SIZE_MAX) const;

        void Upload(vk::CommandBuffer cmdBuffer) { StagedMemory::Upload(cmdBuffer); }
        void Download(vk::CommandBuffer cmdBuffer) { StagedMemory::Download(cmdBuffer, { 0, 0, GetUsedByteSize() }); }
        void DownloadRange(vk::CommandBuffer cmdBuffer, size_t startIndex, size_t count);

        constexpr size_t GetSize() const noexcept { return m_size; }
        constexpr size_t GetCapacity() const noexcept { return m_capacity; }
        constexpr bool IsEmpty() const noexcept { return m_size == 0; }
        constexpr bool IsFull() const noexcept { return m_size == m_capacity; }

        constexpr size_t GetElementSize() const noexcept { return sizeof(DATA); }
        constexpr vk::DeviceSize GetUsedByteSize() const noexcept { return m_size * sizeof(DATA); }
        constexpr vk::DeviceSize GetUnusedByteSize() const noexcept { return (m_capacity - m_size) * sizeof(DATA); }

        constexpr double GetMemoryUsageMB() const noexcept
        {
            return static_cast<double>(GetByteLength()) / (1024.0 * 1024.0);
        }
        constexpr double GetMemoryEfficiency() const noexcept
        {
            return m_capacity > 0 ? static_cast<double>(m_size) / static_cast<double>(m_capacity) : 0.0;
        }

        void Fill(const DATA& value);
        void FillAndUpload(vk::CommandBuffer cmdBuffer, const DATA& value);

        bool TryGetElement(size_t index, DATA& dst) const;
        bool TrySetElement(size_t index, const DATA& value);

        void LogArrayInfo() const;

    private:
        size_t m_size;
        size_t m_capacity;

        size_t CalculateGrowthCapacity(size_t requestedSize) const;
        void AllocateIfNeeded(size_t requiredCapacity);
        void ReallocateIfNeeded(size_t requiredCapacity);
        void Allocate(size_t capacity);
        void Reallocate(size_t capacity);

        bool ValidateIndex(size_t index) const;
        bool ValidateRange(size_t startIndex, size_t count) const;
        bool ValidateInsertIndex(size_t index) const;

        constexpr vk::DeviceSize GetElementOffset(size_t index) const noexcept
        {
            return static_cast<vk::DeviceSize>(index * sizeof(DATA));
        }
        constexpr vk::DeviceSize GetElementSizeBytes(size_t count) const noexcept
        {
            return static_cast<vk::DeviceSize>(count * sizeof(DATA));
        }
    };

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetSize(size_t size)
    {
        if (size == m_size)
        {
            return;
        }

        AllocateIfNeeded(size);
        m_size = size;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Resize(size_t size)
    {
        if (size == m_size)
        {
            return;
        }

        ReallocateIfNeeded(size);
        m_size = size;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Resize(size_t size, const DATA& value)
    {
        size_t oldSize = m_size;
        Resize(size);

        if (size > oldSize)
        {
            SetRange(oldSize, size - oldSize, value);
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Reserve(size_t capacity)
    {
        if (capacity > m_capacity)
        {
            Reallocate(capacity);
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::ShrinkToFit()
    {
        if (m_size < m_capacity)
        {
            Reallocate(m_size);
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::PushBack(const DATA& value)
    {
        ReallocateIfNeeded(m_size + 1);
        SetElement(m_size, value);
        ++m_size;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::PushBackAndUpload(vk::CommandBuffer cmdBuffer, const DATA& value)
    {
        PushBack(value);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::PopBack()
    {
        if (m_size > 0)
        {
            --m_size;
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Insert(size_t index, const DATA& value)
    {
        if (!ValidateInsertIndex(index)) return;

        ReallocateIfNeeded(m_size + 1);

        if (index < m_size)
        {
            std::vector<DATA> temp;
            GetElements(index, m_size - index, temp);
            SetElements(index + 1, temp);
        }

        SetElement(index, value);
        ++m_size;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Insert(size_t index, const std::vector<DATA>& values)
    {
        Insert(index, values.data(), values.size());
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Insert(size_t index, const DATA* values, size_t count)
    {
        if (!ValidateInsertIndex(index) || !values || count == 0) return;

        ReallocateIfNeeded(m_size + count);

        if (index < m_size)
        {
            std::vector<DATA> temp;
            GetElements(index, m_size - index, temp);
            SetElements(index + count, temp);
        }

        SetElements(index, values, count);
        m_size += count;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Erase(size_t index)
    {
        Erase(index, 1);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Erase(size_t startIndex, size_t count)
    {
        if (!ValidateRange(startIndex, count)) return;

        if (startIndex + count < m_size)
        {
            std::vector<DATA> temp;
            GetElements(startIndex + count, m_size - startIndex - count, temp);
            SetElements(startIndex, temp);
        }

        m_size -= count;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Assign(const std::vector<DATA>& src)
    {
        AllocateIfNeeded(src.size());
        m_size = src.size();
        StagedMemory::CopyFromHost(src.data(), { 0, 0, GetElementSizeBytes(src.size()) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Assign(const DATA* src, size_t count)
    {
        if (!src)
        {
            mcore::Logger::Error("Source pointer is null in Assign");
            return;
        }

        AllocateIfNeeded(count);
        m_size = count;
        StagedMemory::CopyFromHost(src, { 0, 0, GetElementSizeBytes(count) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Assign(std::span<const DATA> src)
    {
        Assign(src.data(), src.size());
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Assign(std::initializer_list<DATA> src)
    {
        Assign(src.begin(), src.size());
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetElement(size_t index, const DATA& value)
    {
        if (!ValidateIndex(index)) return;
        StagedMemory::CopyFromHost(&value, { GetElementOffset(index), 0, sizeof(DATA) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::GetElement(size_t index, DATA& dst) const
    {
        if (!ValidateIndex(index)) return;
        StagedMemory::CopyToHost(&dst, { GetElementOffset(index), 0, sizeof(DATA) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetElements(size_t startIndex, const std::vector<DATA>& values)
    {
        SetElements(startIndex, values.data(), values.size());
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetElements(size_t startIndex, const DATA* values, size_t count)
    {
        if (!values || count == 0)
        {
            mcore::Logger::Error("Invalid parameters in SetElements");
            return;
        }

        if (!ValidateRange(startIndex, count)) return;

        StagedMemory::CopyFromHost(values, { GetElementOffset(startIndex), 0, GetElementSizeBytes(count) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetElements(size_t startIndex, std::span<const DATA> values)
    {
        SetElements(startIndex, values.data(), values.size());
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetRange(size_t startIndex, size_t count, const DATA& value)
    {
        if (!ValidateRange(startIndex, count)) return;

        std::vector<DATA> temp(count, value);
        SetElements(startIndex, temp.data(), count);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::GetElements(size_t startIndex, size_t count, std::vector<DATA>& dst) const
    {
        if (!ValidateRange(startIndex, count)) return;

        dst.resize(count);
        StagedMemory::CopyToHost(dst.data(), { GetElementOffset(startIndex), 0, GetElementSizeBytes(count) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::GetElements(size_t startIndex, size_t count, DATA* dst) const
    {
        if (!dst)
        {
            mcore::Logger::Error("Destination pointer is null in GetElements");
            return;
        }

        if (!ValidateRange(startIndex, count)) return;

        StagedMemory::CopyToHost(dst, { GetElementOffset(startIndex), 0, GetElementSizeBytes(count) });
    }

    template<typename DATA, bool CUDA_EXPORT>
    std::vector<DATA> StagedBuffer<DATA, CUDA_EXPORT>::ToVector() const
    {
        std::vector<DATA> result;
        if (m_size > 0)
        {
            GetElements(0, m_size, result);
        }
        return result;
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::AssignAndUpload(vk::CommandBuffer cmdBuffer, const std::vector<DATA>& src)
    {
        Assign(src);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::AssignAndUpload(vk::CommandBuffer cmdBuffer, const DATA* src, size_t count)
    {
        Assign(src, count);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::AssignAndUpload(vk::CommandBuffer cmdBuffer, std::span<const DATA> src)
    {
        Assign(src);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::AssignAndUpload(vk::CommandBuffer cmdBuffer, std::initializer_list<DATA> src)
    {
        Assign(src);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetElementAndUpload(vk::CommandBuffer cmdBuffer, size_t index, const DATA& value)
    {
        SetElement(index, value);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::SetElementsAndUpload(vk::CommandBuffer cmdBuffer, size_t startIndex, const std::vector<DATA>& values)
    {
        SetElements(startIndex, values);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::DownloadAndGetElement(vk::CommandBuffer cmdBuffer, size_t index, DATA& dst)
    {
        DownloadRange(cmdBuffer, index, 1);
        GetElement(index, dst);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::DownloadAndGetElements(vk::CommandBuffer cmdBuffer, size_t startIndex, size_t count, std::vector<DATA>& dst)
    {
        DownloadRange(cmdBuffer, startIndex, count);
        GetElements(startIndex, count, dst);
    }

    template<typename DATA, bool CUDA_EXPORT>
    std::vector<DATA> StagedBuffer<DATA, CUDA_EXPORT>::DownloadToVector(vk::CommandBuffer cmdBuffer)
    {
        Download(cmdBuffer);
        return ToVector();
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::CopyFromArray(vk::CommandBuffer cmdBuffer, const StagedBuffer<DATA, CUDA_EXPORT>& src,
        size_t srcIndex, size_t dstIndex, size_t count)
    {
        if (count == SIZE_MAX)
        {
            count = std::min(src.GetSize() - srcIndex, GetSize() - dstIndex);
        }

        if (count == 0)
        {
            mcore::Logger::Debug("Copy count is zero, skipping operation");
            return;
        }

        if (!ValidateRange(dstIndex, count))
        {
            mcore::Logger::Error("Destination range validation failed");
            return;
        }

        if (!src.ValidateRange(srcIndex, count))
        {
            mcore::Logger::Error("Source range validation failed");
            return;
        }

        vk::BufferCopy copyRegion{
            src.GetElementOffset(srcIndex),
            GetElementOffset(dstIndex),
            GetElementSizeBytes(count)
        };

        StagedMemory::CopyFromDevice(cmdBuffer, src, copyRegion);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::CopyToArray(vk::CommandBuffer cmdBuffer, StagedBuffer<DATA, CUDA_EXPORT>& dst,
        size_t srcIndex, size_t dstIndex, size_t count) const
    {
        dst.CopyFromArray(cmdBuffer, *this, srcIndex, dstIndex, count);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::DownloadRange(vk::CommandBuffer cmdBuffer, size_t startIndex, size_t count)
    {
        if (!ValidateRange(startIndex, count)) return;

        StagedMemory::Download(cmdBuffer,
            {
                GetElementOffset(startIndex),
                GetElementOffset(startIndex),
                GetElementSizeBytes(count)
            });
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Fill(const DATA& value)
    {
        if (m_size > 0)
        {
            SetRange(0, m_size, value);
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::FillAndUpload(vk::CommandBuffer cmdBuffer, const DATA& value)
    {
        Fill(value);
        Upload(cmdBuffer);
    }

    template<typename DATA, bool CUDA_EXPORT>
    bool StagedBuffer<DATA, CUDA_EXPORT>::TryGetElement(size_t index, DATA& dst) const
    {
        if (!ValidateIndex(index)) return false;

        try
        {
            GetElement(index, dst);
            return true;
        }
        catch (const std::exception& e)
        {
            mcore::Logger::Error("Exception in TryGetElement: ", e.what());
            return false;
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    bool StagedBuffer<DATA, CUDA_EXPORT>::TrySetElement(size_t index, const DATA& value)
    {
        if (!ValidateIndex(index)) return false;

        try
        {
            SetElement(index, value);
            return true;
        }
        catch (const std::exception& e)
        {
            mcore::Logger::Error("Exception in TrySetElement: ", e.what());
            return false;
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::LogArrayInfo() const
    {
        mcore::Logger::Debug("StagedBuffer<", typeid(DATA).name(), "> Info:");
        mcore::Logger::Debug("  Size: ", m_size, " / ", m_capacity, " elements");
        mcore::Logger::Debug("  Element Size: ", sizeof(DATA), " bytes");
        mcore::Logger::Debug("  Used Memory: ", GetUsedByteSize(), " bytes");
        mcore::Logger::Debug("  Total Memory: ", GetByteLength(), " bytes (", GetMemoryUsageMB(), " MB)");
        mcore::Logger::Debug("  Memory Efficiency: ", (GetMemoryEfficiency() * 100.0), "%");
        mcore::Logger::Debug("  Empty: ", IsEmpty() ? "Yes" : "No");
        mcore::Logger::Debug("  Full: ", IsFull() ? "Yes" : "No");
        mcore::Logger::Debug("  Dirty: ", IsDirty() ? "Yes" : "No");
    }

    template<typename DATA, bool CUDA_EXPORT>
    size_t StagedBuffer<DATA, CUDA_EXPORT>::CalculateGrowthCapacity(size_t requestedSize) const
    {
        if (requestedSize <= m_capacity)
        {
            return m_capacity;
        }

        constexpr size_t MAX_CAPACITY = SIZE_MAX / sizeof(DATA);

        size_t growthCapacity = m_capacity * 3 >> 1;
        size_t newCapacity = std::max(requestedSize, growthCapacity);
        newCapacity = std::min(newCapacity, MAX_CAPACITY);
        return newCapacity;
    }

    template<typename DATA, bool CUDA_EXPORT>
    inline void StagedBuffer<DATA, CUDA_EXPORT>::AllocateIfNeeded(size_t requiredCapacity)
    {
        if (requiredCapacity <= m_capacity)
        {
            return;
        }

        size_t newCapacity = CalculateGrowthCapacity(requiredCapacity);
        Allocate(newCapacity);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::ReallocateIfNeeded(size_t requiredCapacity)
    {
        if (requiredCapacity <= m_capacity)
        {
            return;
        }

        size_t newCapacity = CalculateGrowthCapacity(requiredCapacity);
        Reallocate(newCapacity);
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Allocate(size_t capacity)
    {
        if (capacity == m_capacity)
        {
            return;
        }

        StagedMemory::Destroy();
        m_capacity = capacity;
        if constexpr (CUDA_EXPORT == false)
        {
            StagedMemory::Create(sizeof(DATA) * m_capacity);
        }
        else
        {
            StagedMemory::CreateWithExport(sizeof(DATA) * m_capacity);
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    void StagedBuffer<DATA, CUDA_EXPORT>::Reallocate(size_t capacity)
    {
        if (capacity == m_capacity)
        {
            return;
        }

        std::vector<DATA> backup;
        if (m_size > 0)
        {
            backup.resize(m_size);
            GetElements(0, m_size, backup.data());
        }

        StagedMemory::Destroy();
        m_capacity = capacity;
        if constexpr (CUDA_EXPORT == false)
        {
            StagedMemory::Create(sizeof(DATA) * m_capacity);
        }
        else
        {
            StagedMemory::CreateWithExport(sizeof(DATA) * m_capacity);
        }

        if (!backup.empty())
        {
            StagedMemory::CopyFromHost(backup.data(), { 0, 0, GetElementSizeBytes(backup.size()) });
        }
    }

    template<typename DATA, bool CUDA_EXPORT>
    bool StagedBuffer<DATA, CUDA_EXPORT>::ValidateIndex(size_t index) const
    {
        if (index >= m_size)
        {
            mcore::Logger::Error("Index out of bounds: ", index, " >= ", m_size);
            return false;
        }
        return true;
    }

    template<typename DATA, bool CUDA_EXPORT>
    bool StagedBuffer<DATA, CUDA_EXPORT>::ValidateRange(size_t startIndex, size_t count) const
    {
        if (startIndex + count > m_size)
        {
            mcore::Logger::Error("Range out of bounds: [", startIndex, ", ", startIndex + count, ") > ", m_size);
            return false;
        }
        return true;
    }

    template<typename DATA, bool CUDA_EXPORT>
    bool StagedBuffer<DATA, CUDA_EXPORT>::ValidateInsertIndex(size_t index) const
    {
        if (index > m_size)
        {
            mcore::Logger::Error("Insert index out of bounds: ", index, " > ", m_size);
            return false;
        }
        return true;
    }
}