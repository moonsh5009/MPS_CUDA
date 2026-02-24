#include "stdafx.h"
#include "Texture.h"

#include "VulkanCore.h"

mvk::Texture::~Texture()
{
    Destroy();
}

mvk::Texture::Texture(Texture&& other) noexcept
{
    *this = std::move(other);
}

mvk::Texture& mvk::Texture::operator=(Texture&& other) noexcept
{
    if (this != &other)
    {
        m_image = other.m_image;
        m_imageMemory = other.m_imageMemory;
        m_imageView = other.m_imageView;
        m_stagingBuffer = other.m_stagingBuffer;
        m_stagingMemory = other.m_stagingMemory;
        m_extent = other.m_extent;
        m_format = other.m_format;
        m_usage = other.m_usage;
        m_samples = other.m_samples;
        m_mipLevels = other.m_mipLevels;
        m_arrayLayers = other.m_arrayLayers;
        m_byteSize = other.m_byteSize;
        m_isCreated = other.m_isCreated;
        m_currentLayout = other.m_currentLayout;
        other.m_image = nullptr;
        other.m_imageMemory = nullptr;
        other.m_imageView = nullptr;
        other.m_stagingBuffer = nullptr;
        other.m_stagingMemory = nullptr;
        other.m_byteSize = 0;
        other.m_isCreated = false;
        other.m_currentLayout = vk::ImageLayout::eUndefined;
    }
    return *this;
}

void mvk::Texture::Create(
    vk::Format format,
    vk::ImageUsageFlags usage,
    vk::SampleCountFlagBits samples,
    vk::ImageLayout initialLayout,
    uint32_t mipLevels,
    uint32_t arrayLayers)
{
    m_format = format;
    m_usage = usage;
    m_samples = samples;
    m_mipLevels = mipLevels;
    m_arrayLayers = arrayLayers;
    m_currentLayout = initialLayout;
}

void mvk::Texture::Create(
    const vk::Extent3D& extent,
    vk::Format format,
    vk::ImageUsageFlags usage,
    vk::SampleCountFlagBits samples,
    vk::ImageLayout initialLayout,
    uint32_t mipLevels,
    uint32_t arrayLayers)
{
    m_format = format;
    m_usage = usage;
    m_samples = samples;
    m_mipLevels = mipLevels;
    m_arrayLayers = arrayLayers;
    m_currentLayout = initialLayout;
    Resize(extent);
}

void mvk::Texture::Destroy()
{
    const auto pCore = VulkanCore::Instance();

    if (m_imageView)
    {
        pCore->GetDevice().destroyImageView(m_imageView);
        m_imageView = nullptr;
    }
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
    if (m_image)
    {
        pCore->GetDevice().destroyImage(m_image);
        m_image = nullptr;
    }
    if (m_imageMemory)
    {
        pCore->GetDevice().freeMemory(m_imageMemory);
        m_imageMemory = nullptr;
    }

    m_byteSize = 0;
    m_isCreated = false;
    m_currentLayout = vk::ImageLayout::eUndefined;
}

void mvk::Texture::Resize(const vk::Extent3D& extent)
{
    const auto pCore = VulkanCore::Instance();

    if (m_isCreated)
    {
        Destroy();
    }

    m_extent = extent;

    try
    {
        const auto formatSize = GetFormatSize(m_format);
        m_byteSize = static_cast<vk::DeviceSize>(extent.width) * extent.height * extent.depth * formatSize * m_arrayLayers;

        m_image = pCore->GetDevice().createImage(
            {
                {}, vk::ImageType::e2D, m_format, extent,
                m_mipLevels, m_arrayLayers, m_samples,
                vk::ImageTiling::eOptimal,
                m_usage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
                vk::SharingMode::eExclusive,
                {}, {},
                m_currentLayout
            });

        const auto memReq = pCore->GetDevice().getImageMemoryRequirements(m_image);
        const auto memType = pCore->FindMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        m_imageMemory = pCore->GetDevice().allocateMemory({ memReq.size, memType });
        pCore->GetDevice().bindImageMemory(m_image, m_imageMemory, 0);

        vk::ImageViewType viewType = vk::ImageViewType::e2D;
        if (m_arrayLayers == 6)
            viewType = vk::ImageViewType::eCube;
        else if (m_arrayLayers > 1)
            viewType = vk::ImageViewType::e2DArray;

        const auto aspectMask = DetermineAspectMask();
        m_imageView = pCore->GetDevice().createImageView({
            {}, m_image,
            viewType,
            m_format,
            {},
            { aspectMask, 0, m_mipLevels, 0, m_arrayLayers }
            });

        m_stagingBuffer = pCore->GetDevice().createBuffer({
            {}, m_byteSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive
            });

        const auto stagingReq = pCore->GetDevice().getBufferMemoryRequirements(m_stagingBuffer);
        const auto stagingType = pCore->FindMemoryType(stagingReq.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        m_stagingMemory = pCore->GetDevice().allocateMemory({ stagingReq.size, stagingType });
        pCore->GetDevice().bindBufferMemory(m_stagingBuffer, m_stagingMemory, 0);

        m_isCreated = true;
    }
    catch (const vk::SystemError& e)
    {
        Destroy();
        throw std::runtime_error("Failed to create Texture: " + std::string(e.what()));
    }
}

void mvk::Texture::CopyFromHost(const void* src, uint32_t layerIndex)
{
    if (!IsValid()) return;

    const auto pCore = VulkanCore::Instance();

    const vk::DeviceSize layerSize = CalculateLayerSize();
    const vk::DeviceSize offset = (layerIndex == UINT32_MAX) ? 0 : layerIndex * layerSize;
    const vk::DeviceSize copySize = (layerIndex == UINT32_MAX) ? m_byteSize : layerSize;

    const auto mappedStaging = pCore->GetDevice().mapMemory(m_stagingMemory, offset, copySize);
    if (mappedStaging)
    {
        std::memcpy(mappedStaging, src, static_cast<size_t>(copySize));
        pCore->GetDevice().unmapMemory(m_stagingMemory);
    }
}

void mvk::Texture::CopyToHost(void* dst, uint32_t layerIndex) const
{
    if (!IsValid()) return;

    const auto pCore = VulkanCore::Instance();

    const auto layerSize = CalculateLayerSize();
    const auto offset = layerIndex == UINT32_MAX ? 0 : layerIndex * layerSize;
    const auto copySize = layerIndex == UINT32_MAX ? m_byteSize : layerSize;

    const auto mappedStaging = pCore->GetDevice().mapMemory(m_stagingMemory, offset, copySize);
    if (mappedStaging)
    {
        std::memcpy(dst, mappedStaging, static_cast<size_t>(copySize));
        pCore->GetDevice().unmapMemory(m_stagingMemory);
    }
}

void mvk::Texture::CopyToHost(void* dst, uint32_t x, uint32_t y, uint32_t layerIndex) const
{
    if (!IsValid()) return;

    const auto pCore = VulkanCore::Instance();

    const auto formatSize = GetFormatSize(m_format);
    const auto layerSize = CalculateLayerSize();

    const auto offset = static_cast<vk::DeviceSize>(y * m_extent.width + x) * formatSize +
        (layerIndex == UINT32_MAX ? 0 : layerIndex * layerSize);
    const auto copySize = formatSize;

    const auto mappedStaging = pCore->GetDevice().mapMemory(m_stagingMemory, offset, copySize);
    if (mappedStaging)
    {
        std::memcpy(dst, mappedStaging, static_cast<size_t>(copySize));
        pCore->GetDevice().unmapMemory(m_stagingMemory);
    }
}

void mvk::Texture::Upload(vk::CommandBuffer cmdBuffer, uint32_t layerIndex) const
{
    if (!IsValid()) return;

    const auto startLayer = (layerIndex == UINT32_MAX) ? 0 : layerIndex;
    const auto layerCount = (layerIndex == UINT32_MAX) ? m_arrayLayers : 1;
    const auto layerSize = CalculateLayerSize();

    for (uint32_t layer = startLayer; layer < startLayer + layerCount; ++layer)
    {
        const auto bufferOffset = layer * layerSize;
        vk::BufferImageCopy region = {
            bufferOffset, 0, 0,
            { DetermineAspectMask(), 0, layer, 1 },
            { 0, 0, 0 },
            m_extent
        };
        cmdBuffer.copyBufferToImage(m_stagingBuffer, m_image, m_currentLayout, region);
    }
}

void mvk::Texture::Download(vk::CommandBuffer cmdBuffer, uint32_t layerIndex) const
{
    if (!IsValid()) return;

    const auto startLayer = (layerIndex == UINT32_MAX) ? 0 : layerIndex;
    const auto layerCount = (layerIndex == UINT32_MAX) ? m_arrayLayers : 1;
    const auto layerSize = CalculateLayerSize();

    for (uint32_t layer = startLayer; layer < startLayer + layerCount; ++layer)
    {
        const auto bufferOffset = layer * layerSize;
        vk::BufferImageCopy region = {
            bufferOffset, 0, 0,
            { DetermineAspectMask(), 0, layer, 1 },
            { 0, 0, 0 },
            m_extent
        };
        cmdBuffer.copyImageToBuffer(m_image, m_currentLayout, m_stagingBuffer, region);
    }
}

void mvk::Texture::Download(vk::CommandBuffer cmdBuffer, uint32_t x, uint32_t y, uint32_t layerIndex) const
{
    if (!IsValid()) return;

    const auto formatSize = GetFormatSize(m_format);
    const auto offset = static_cast<vk::DeviceSize>(y * m_extent.width + x) * formatSize +
        (layerIndex == UINT32_MAX ? 0 : layerIndex * CalculateLayerSize());
    vk::BufferImageCopy region = {
        offset, 0, 0,
        { DetermineAspectMask(), 0, layerIndex == UINT32_MAX ? 0 : layerIndex, 1 },
        { static_cast<int32_t>(x), static_cast<int32_t>(y), 0 },
        { 1, 1, 1 }
    };
	cmdBuffer.copyImageToBuffer(m_image, m_currentLayout, m_stagingBuffer, region);
}

void mvk::Texture::TransitionLayout(vk::CommandBuffer cmdBuffer, vk::ImageLayout newLayout)
{
    if (!IsValid() || m_currentLayout == newLayout) return;

    const auto transitionInfo = DetermineLayoutTransition(m_currentLayout, newLayout);
    const auto aspectMask = DetermineAspectMask();

    TransitionLayout(cmdBuffer, m_currentLayout, newLayout,
        transitionInfo.srcStage, transitionInfo.dstStage, aspectMask,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
}

void mvk::Texture::TransitionLayout(
    vk::CommandBuffer cmdBuffer,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    vk::PipelineStageFlags srcStage,
    vk::PipelineStageFlags dstStage,
    vk::ImageAspectFlags aspectMask,
    uint32_t srcQueueFamilyIndex,
    uint32_t dstQueueFamilyIndex)
{
    if (!IsValid()) return;

    const auto transitionInfo = DetermineLayoutTransition(oldLayout, newLayout);

    vk::ImageMemoryBarrier barrier = {
        transitionInfo.srcAccess, transitionInfo.dstAccess,
        oldLayout, newLayout,
        srcQueueFamilyIndex, dstQueueFamilyIndex,
        m_image,
        { aspectMask, 0, m_mipLevels, 0, m_arrayLayers }
    };

    cmdBuffer.pipelineBarrier(srcStage, dstStage, {}, nullptr, nullptr, barrier);
    m_currentLayout = newLayout;
}

void mvk::Texture::GenerateMipmaps(vk::CommandBuffer cmdBuffer)
{
    if (!IsValid() || m_mipLevels <= 1) return;

    const auto pCore = VulkanCore::Instance();

    const vk::ImageAspectFlags aspectMask = DetermineAspectMask();

    vk::ImageMemoryBarrier barrier{};
    barrier.image = m_image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = m_arrayLayers;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = static_cast<int32_t>(m_extent.width);
    int32_t mipHeight = static_cast<int32_t>(m_extent.height);

    for (uint32_t i = 1; i < m_mipLevels; ++i)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eTransfer,
            {}, nullptr, nullptr, barrier);

        vk::ImageBlit blit{};
        blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
        blit.srcOffsets[1] = vk::Offset3D{ mipWidth, mipHeight, 1 };
        blit.srcSubresource = { aspectMask, i - 1, 0, m_arrayLayers };
        blit.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
        blit.dstOffsets[1] = vk::Offset3D{
            mipWidth > 1 ? mipWidth / 2 : 1,
            mipHeight > 1 ? mipHeight / 2 : 1,
            1
        };
        blit.dstSubresource = { aspectMask, i, 0, m_arrayLayers };

        cmdBuffer.blitImage(
            m_image, vk::ImageLayout::eTransferSrcOptimal,
            m_image, vk::ImageLayout::eTransferDstOptimal,
            blit, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {}, nullptr, nullptr, barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = m_mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    cmdBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, nullptr, nullptr, barrier);

    m_currentLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
}

vk::ImageAspectFlags mvk::Texture::DetermineAspectMask() const
{
    if (IsDepthStencilFormat(m_format))
    {
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    }
    if (IsDepthFormat(m_format))
    {
        return vk::ImageAspectFlagBits::eDepth;
    }
    if (IsStencilFormat(m_format))
    {
        return vk::ImageAspectFlagBits::eStencil;
    }
    return vk::ImageAspectFlagBits::eColor;
}

mvk::Texture::LayoutTransitionInfo mvk::Texture::DetermineLayoutTransition(
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout) const
{
    LayoutTransitionInfo info{};

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        info.dstStage = vk::PipelineStageFlagBits::eTransfer;
        info.srcAccess = {};
        info.dstAccess = vk::AccessFlagBits::eTransferWrite;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eTransfer;
        info.dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.srcAccess = vk::AccessFlagBits::eTransferWrite;
        info.dstAccess = vk::AccessFlagBits::eShaderRead;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eGeneral)
    {
        info.srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        info.dstStage = vk::PipelineStageFlagBits::eComputeShader;
        info.srcAccess = {};
        info.dstAccess = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        info.dstStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        info.srcAccess = {};
        info.dstAccess = vk::AccessFlagBits::eColorAttachmentWrite;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        info.dstStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        info.srcAccess = {};
        info.dstAccess = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    }
    else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        info.dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.srcAccess = vk::AccessFlagBits::eColorAttachmentWrite;
        info.dstAccess = vk::AccessFlagBits::eShaderRead;
    }
    else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eLateFragmentTests;
        info.dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.srcAccess = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        info.dstAccess = vk::AccessFlagBits::eShaderRead;
    }
    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eColorAttachmentOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.dstStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        info.srcAccess = vk::AccessFlagBits::eShaderRead;
        info.dstAccess = vk::AccessFlagBits::eColorAttachmentWrite;
    }
    else if (oldLayout == vk::ImageLayout::eTransferSrcOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eTransfer;
        info.dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.srcAccess = vk::AccessFlagBits::eTransferRead;
        info.dstAccess = vk::AccessFlagBits::eShaderRead;
    }
    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.dstStage = vk::PipelineStageFlagBits::eTransfer;
        info.srcAccess = vk::AccessFlagBits::eShaderRead;
        info.dstAccess = vk::AccessFlagBits::eTransferRead;
    }
    else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        info.srcStage = vk::PipelineStageFlagBits::eFragmentShader;
        info.dstStage = vk::PipelineStageFlagBits::eTransfer;
        info.srcAccess = vk::AccessFlagBits::eShaderRead;
        info.dstAccess = vk::AccessFlagBits::eTransferWrite;
    }
    else
    {
        info.srcStage = vk::PipelineStageFlagBits::eAllCommands;
        info.dstStage = vk::PipelineStageFlagBits::eAllCommands;
        info.srcAccess = {};
        info.dstAccess = {};
    }

    return info;
}

bool mvk::Texture::IsDepthFormat(vk::Format format) const
{
    switch (format)
    {
    case vk::Format::eD16Unorm:
    case vk::Format::eD32Sfloat:
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return true;
    default:
        return false;
    }
}

bool mvk::Texture::IsStencilFormat(vk::Format format) const
{
    switch (format)
    {
    case vk::Format::eS8Uint:
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return true;
    default:
        return false;
    }
}

bool mvk::Texture::IsDepthStencilFormat(vk::Format format) const
{
    switch (format)
    {
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return true;
    default:
        return false;
    }
}

vk::DeviceSize mvk::Texture::CalculateLayerSize() const
{
    if (m_arrayLayers == 0) return 0;
    return m_byteSize / m_arrayLayers;
}