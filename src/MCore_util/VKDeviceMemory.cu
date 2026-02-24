#include "stdafx.h"
#include "VKDeviceMemory.h"

#include "MCudaUtil.cuh"

#include "VulkanCore.h"

using namespace mcuda;

mcuda::VKDeviceMemory::~VKDeviceMemory()
{
    VKDeviceMemory::Destroy();
}

VKDeviceMemory::VKDeviceMemory(VKDeviceMemory&& src) noexcept
{
    *this = std::move(src);
}

VKDeviceMemory& VKDeviceMemory::operator=(VKDeviceMemory&& src) noexcept
{
    if (this != &src)
    {
        VKDeviceMemory::Destroy();

        IDeviceMemory::operator=(std::move(src));

        m_usage = src.m_usage;
        m_memory = std::exchange(src.m_memory, nullptr);
        m_buffer = std::exchange(src.m_buffer, nullptr);
        m_memoryHandle = src.m_memoryHandle;
        m_cudaExtMem = src.m_cudaExtMem;

    #ifdef _WIN32
        src.m_memoryHandle = nullptr;
    #else
        src.m_memoryHandle = -1;
    #endif
        src.m_cudaExtMem = nullptr;
    }
    return *this;
}

void VKDeviceMemory::Initialize(vk::BufferUsageFlags usage)
{
    m_usage = usage;
}

void VKDeviceMemory::Create(size_t byteSize)
{
    Destroy();

    m_byteSize = byteSize;
    if (byteSize == 0)
        return;

    CreateVulkanBuffer(byteSize);
    ImportFromVulkan();
}

void VKDeviceMemory::Destroy()
{
    const auto pCore = mvk::VulkanCore::Instance();

    try
    {
        if (m_rawPtr)
        {
            m_rawPtr = nullptr;
        }

        if (m_cudaExtMem)
        {
            CUDA_CHECK(cudaDestroyExternalMemory(static_cast<cudaExternalMemory_t>(m_cudaExtMem)));
            m_cudaExtMem = nullptr;
        }

    #ifdef _WIN32
        if (m_memoryHandle != nullptr)
        {
            CloseHandle(m_memoryHandle);
            m_memoryHandle = nullptr;
        }
    #else
        if (m_memoryHandle != -1)
        {
            close(m_memoryHandle);
            m_memoryHandle = -1;
        }
    #endif

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
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error during Destroy: ", e.what());
        mcore::Logger::Print();
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error during Destroy: ", e.what());
        mcore::Logger::Print();
    }
}

void VKDeviceMemory::CreateVulkanBuffer(size_t byteSize)
{
    const auto pCore = mvk::VulkanCore::Instance();

    try
    {
        vk::ExternalMemoryBufferCreateInfo externalBufferInfo{};
    #ifdef _WIN32
        externalBufferInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
    #else
        externalBufferInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    #endif

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = byteSize;
        bufferInfo.usage = m_usage | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        bufferInfo.pNext = &externalBufferInfo;

        m_buffer = pCore->GetDevice().createBuffer(bufferInfo);

        auto memReq = pCore->GetDevice().getBufferMemoryRequirements(m_buffer);
        auto memoryType = pCore->FindMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

        vk::ExportMemoryAllocateInfo exportAllocInfo{};
    #ifdef _WIN32
        exportAllocInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
    #else
        exportAllocInfo.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
    #endif

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryType;
        allocInfo.pNext = &exportAllocInfo;

        m_memory = pCore->GetDevice().allocateMemory(allocInfo);
        pCore->GetDevice().bindBufferMemory(m_buffer, m_memory, 0);
    }
    catch (const vk::SystemError& e)
    {
        mcore::Logger::Error("Vulkan error in CreateWithExport: ", e.what());
        mcore::Logger::Print();
        Destroy();
        throw;
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Error("Error in CreateWithExport: ", e.what());
        mcore::Logger::Print();
        Destroy();
        throw;
    }
}

void VKDeviceMemory::ImportFromVulkan()
{
    const auto pCore = mvk::VulkanCore::Instance();
    try
    {
        m_memoryHandle = pCore->GetVulkanMemoryHandle(m_memory);
        if (m_memoryHandle == nullptr)
        {
            mcore::Logger::Debug("Failed to get Win32 handle");
            mcore::Logger::Print();
            Destroy();
            return;
        }
    }
    catch (const std::exception& e)
    {
        mcore::Logger::Debug("Failed to get memory handle: ", e.what());
        mcore::Logger::Print();
        Destroy();
        return;
    }

    cudaExternalMemoryHandleDesc memHandleDesc{};
    memset(&memHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
#ifdef _WIN32
    memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memHandleDesc.handle.win32.handle = m_memoryHandle;
#else
    memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memHandleDesc.handle.fd.handle = m_memoryHandle;
#endif
    memHandleDesc.size = m_byteSize;

    CUDA_CHECK(cudaImportExternalMemory(reinterpret_cast<cudaExternalMemory_t*>(&m_cudaExtMem), &memHandleDesc));

    cudaExternalMemoryBufferDesc bufferDesc{};
    memset(&bufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = m_byteSize;
    bufferDesc.flags = 0;

    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&m_rawPtr,
        static_cast<cudaExternalMemory_t>(m_cudaExtMem),
        &bufferDesc));

    mcore::Logger::Debug("Successfully created shared Vulkan-CUDA buffer!");
    mcore::Logger::Debug("Vulkan buffer: ", static_cast<VkBuffer>(m_buffer));
    mcore::Logger::Debug("CUDA device ptr: ", m_rawPtr);
    mcore::Logger::Print();
}