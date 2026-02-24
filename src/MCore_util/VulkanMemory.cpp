#include "stdafx.h"
#include "VulkanMemory.h"

mvk::VulkanMemory::VulkanMemory()
    : m_buffer{}
    , m_memory{}
    , m_byteSize{ 0 }
    , m_usage{}
    , m_dirtyRange{ 0, 0 }
{}

mvk::VulkanMemory::VulkanMemory(vk::BufferUsageFlags usage)
    : m_buffer{}
    , m_memory{}
    , m_byteSize{ 0 }
    , m_usage{ usage }
    , m_dirtyRange{ 0, 0 }
{}

mvk::VulkanMemory::VulkanMemory(VulkanMemory&& other) noexcept
{
	*this = std::move(other);
}

mvk::VulkanMemory& mvk::VulkanMemory::operator=(VulkanMemory&& other) noexcept
{
    if (this != &other)
    {
        m_buffer = std::exchange(other.m_buffer, nullptr);
        m_memory = std::exchange(other.m_memory, nullptr);
        m_byteSize = std::exchange(other.m_byteSize, 0);
        m_usage = other.m_usage;
        m_dirtyRange = std::exchange(other.m_dirtyRange, { 0, 0 });
    }
    return *this;
}

void mvk::VulkanMemory::Initialize(vk::BufferUsageFlags usage)
{
    m_usage = usage;
}

void mvk::VulkanMemory::CopyFromDevice(vk::CommandBuffer cmdBuffer, const VulkanMemory& src, const vk::BufferCopy& copyRegion)
{
    cmdBuffer.copyBuffer(src.GetBuffer(), GetBuffer(), copyRegion);
    m_dirtyRange.first = std::min(m_dirtyRange.first, copyRegion.dstOffset);
    m_dirtyRange.second = std::max(m_dirtyRange.second, copyRegion.dstOffset + copyRegion.size);
}

void mvk::VulkanMemory::CopyToDevice(vk::CommandBuffer cmdBuffer, VulkanMemory& dst, const vk::BufferCopy& copyRegion) const
{
    dst.CopyFromDevice(cmdBuffer, *this, { copyRegion.dstOffset, copyRegion.srcOffset, copyRegion.size });
}