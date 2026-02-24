#include "stdafx.h"
#include "StagedMemory.h"

#include "VulkanCore.h"

mvk::StagedMemory::StagedMemory(vk::BufferUsageFlags usage)
    : VulkanMemory{ usage }
    , m_stagingBuffer{}
    , m_stagingMemory{}
{}

mvk::StagedMemory::~StagedMemory()
{
    Destroy();
}

mvk::StagedMemory::StagedMemory(StagedMemory&& other) noexcept
{
	*this = std::move(other);
}

mvk::StagedMemory& mvk::StagedMemory::operator=(StagedMemory&& other) noexcept
{
    if (this != &other)
    {
        StagedMemory::Destroy();

        VulkanMemory::operator=(std::move(other));

        m_stagingBuffer = std::exchange(other.m_stagingBuffer, nullptr);
        m_stagingMemory = std::exchange(other.m_stagingMemory, nullptr);
    }
	return *this;
}

void mvk::StagedMemory::Create(vk::DeviceSize byteSize)
{
    const auto pCore = VulkanCore::Instance();

    m_byteSize = byteSize;
    if (m_byteSize == 0) return;

    try
    {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = byteSize;
        bufferInfo.usage = m_usage | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        m_buffer = pCore->GetDevice().createBuffer(bufferInfo);

        auto memReq = pCore->GetDevice().getBufferMemoryRequirements(m_buffer);
        auto memoryType = pCore->FindMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryType;

        m_memory = pCore->GetDevice().allocateMemory(allocInfo);
        pCore->GetDevice().bindBufferMemory(m_buffer, m_memory, 0);

        CreateStagingBuffer(byteSize);

        m_dirtyRange = { 0, m_byteSize };
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error in Create: ", e.what());
        Destroy();
        throw;
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error in Create: ", e.what());
        Destroy();
        throw;
    }
}

void mvk::StagedMemory::CreateWithExport(vk::DeviceSize byteSize)
{
    const auto pCore = VulkanCore::Instance();

    m_byteSize = byteSize;
    if (m_byteSize == 0) return;

    try
    {
        vk::ExternalMemoryBufferCreateInfo externalBufferInfo{};
        externalBufferInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = byteSize;
        bufferInfo.usage = m_usage | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        bufferInfo.pNext = &externalBufferInfo;

        m_buffer = pCore->GetDevice().createBuffer(bufferInfo);

        auto memReq = pCore->GetDevice().getBufferMemoryRequirements(m_buffer);
        auto memoryType = pCore->FindMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

        vk::ExportMemoryAllocateInfo exportAllocInfo{};
        exportAllocInfo.sType = vk::StructureType::eExternalMemoryBufferCreateInfo;
        exportAllocInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryType;
        allocInfo.pNext = &exportAllocInfo;

        m_memory = pCore->GetDevice().allocateMemory(allocInfo);
        pCore->GetDevice().bindBufferMemory(m_buffer, m_memory, 0);

        CreateStagingBuffer(byteSize);

        m_dirtyRange = { 0, m_byteSize };
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error in CreateWithExport: ", e.what());
        Destroy();
        throw;
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error in CreateWithExport: ", e.what());
        Destroy();
        throw;
    }
}

void mvk::StagedMemory::CreateStagingBuffer(vk::DeviceSize byteSize)
{
    const auto pCore = VulkanCore::Instance();

    vk::BufferCreateInfo stagingBufferInfo{};
    stagingBufferInfo.size = byteSize;
    stagingBufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    stagingBufferInfo.sharingMode = vk::SharingMode::eExclusive;

    m_stagingBuffer = pCore->GetDevice().createBuffer(stagingBufferInfo);

    auto stagingMemReq = pCore->GetDevice().getBufferMemoryRequirements(m_stagingBuffer);
    auto stagingMemoryType = pCore->FindMemoryType(
        stagingMemReq.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    vk::MemoryAllocateInfo stagingAllocInfo{};
    stagingAllocInfo.allocationSize = stagingMemReq.size;
    stagingAllocInfo.memoryTypeIndex = stagingMemoryType;

    m_stagingMemory = pCore->GetDevice().allocateMemory(stagingAllocInfo);
    pCore->GetDevice().bindBufferMemory(m_stagingBuffer, m_stagingMemory, 0);
}

void mvk::StagedMemory::Destroy()
{
    const auto pCore = VulkanCore::Instance();

    try
    {
        if (m_stagingBuffer)
        {
            pCore->GetDevice().destroyBuffer(m_stagingBuffer);
            m_stagingBuffer = nullptr;
        }

        if (m_stagingMemory)
        {
            pCore->GetDevice().freeMemory(m_stagingMemory);
            m_stagingMemory = nullptr;
        }

        if (m_buffer)
        {
            pCore->GetDevice().destroyBuffer(m_buffer);
            m_buffer = nullptr;
        }

        if (m_memory)
        {
            pCore->GetDevice().freeMemory(m_memory);
            m_memory = nullptr;
        }

        m_byteSize = 0;
        m_dirtyRange = { 0, 0 };

        mcore::Logger::Debug("StagedMemory destroyed successfully");
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error during Destroy: ", e.what());
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error during Destroy: ", e.what());
    }
}

void mvk::StagedMemory::CopyFromHost(const void* src, const vk::BufferCopy& copyRegion)
{
    const auto pCore = VulkanCore::Instance();

    if (!src)
    {
        mcore::Logger::Error("Source pointer is null in CopyFromHost");
        return;
    }

    if (copyRegion.dstOffset + copyRegion.size > m_byteSize)
    {
        mcore::Logger::Error("Copy region exceeds buffer size. Offset: ", copyRegion.dstOffset,
            ", Size: ", copyRegion.size, ", Buffer size: ", m_byteSize);
        return;
    }

    if (copyRegion.size == 0)
    {
        mcore::Logger::Debug("Zero-size copy operation, skipping");
        return;
    }

    try
    {
        const auto mappedStaging = pCore->GetDevice().mapMemory(
            m_stagingMemory,
            copyRegion.dstOffset,
            copyRegion.size
        );

        if (!mappedStaging)
        {
            mcore::Logger::Error("Failed to map staging memory");
            return;
        }

        std::memcpy(mappedStaging,
            static_cast<const char*>(src) + copyRegion.srcOffset,
            copyRegion.size);

        pCore->GetDevice().unmapMemory(m_stagingMemory);

        m_dirtyRange.first = std::min(m_dirtyRange.first, copyRegion.dstOffset);
        m_dirtyRange.second = std::max(m_dirtyRange.second, copyRegion.dstOffset + copyRegion.size);
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error in CopyFromHost: ", e.what());
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Exception in CopyFromHost: ", e.what());
    }
}

void mvk::StagedMemory::CopyToHost(void* dst, const vk::BufferCopy& copyRegion) const
{
    const auto pCore = VulkanCore::Instance();

    if (!dst)
    {
        mcore::Logger::Error("Destination pointer is null in CopyToHost");
        return;
    }

    if (copyRegion.srcOffset + copyRegion.size > m_byteSize)
    {
        mcore::Logger::Error("Copy region exceeds buffer size. Offset: ", copyRegion.srcOffset,
            ", Size: ", copyRegion.size, ", Buffer size: ", m_byteSize);
        return;
    }

    if (copyRegion.size == 0)
    {
        mcore::Logger::Debug("Zero-size copy operation, skipping");
        return;
    }

    try
    {
        const auto mappedStaging = pCore->GetDevice().mapMemory(
            m_stagingMemory,
            copyRegion.srcOffset,
            copyRegion.size
        );

        if (!mappedStaging)
        {
            mcore::Logger::Error("Failed to map staging memory for reading");
            return;
        }

        std::memcpy(static_cast<char*>(dst) + copyRegion.dstOffset,
            mappedStaging,
            copyRegion.size);

        pCore->GetDevice().unmapMemory(m_stagingMemory);
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error in CopyToHost: ", e.what());
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Exception in CopyToHost: ", e.what());
    }
}

void mvk::StagedMemory::Download(vk::CommandBuffer cmdBuffer, const vk::BufferCopy& region)
{
    if (region.srcOffset + region.size > m_byteSize)
    {
        mcore::Logger::Error("Download region exceeds buffer size. Offset: ", region.srcOffset,
            ", Size: ", region.size, ", Buffer size: ", m_byteSize);
        return;
    }

    try
    {
        cmdBuffer.copyBuffer(m_buffer, m_stagingBuffer, region);
        m_dirtyRange = { m_byteSize, 0 };
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error in Download: ", e.what());
    }
}

void mvk::StagedMemory::Upload(vk::CommandBuffer cmdBuffer)
{
    if (!IsDirty()) return;

    try
    {
        vk::DeviceSize copySize = m_dirtyRange.second - m_dirtyRange.first;
        vk::BufferCopy region{
            m_dirtyRange.first,
            m_dirtyRange.first,
            copySize
        };
        cmdBuffer.copyBuffer(m_stagingBuffer, m_buffer, region);
        m_dirtyRange = { m_byteSize, 0 };
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error in Upload: ", e.what());
    }
}