#include "stdafx.h"
#include "SurfaceTexture.h"

#include "VulkanCore.h"

mvk::SurfaceTexture::SurfaceTexture(
    vk::Image image,
    const vk::Extent2D& extent,
    vk::Format format,
    vk::SampleCountFlagBits samples)
    : m_image{ image }
    , m_format{ format }
    , m_samples{ samples }
    , m_currentLayout{ vk::ImageLayout::eUndefined }
{
    vk::ImageViewCreateInfo viewCreateInfo(
        {},
        image,
        vk::ImageViewType::e2D,
        m_format,
        vk::ComponentMapping(),
        vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor,
            0, 1, 0, 1)
    );
    m_imageView = VulkanCore::Instance()->GetDevice().createImageView(viewCreateInfo);
}

mvk::SurfaceTexture::~SurfaceTexture()
{
    Destroy();
}

mvk::SurfaceTexture::SurfaceTexture(SurfaceTexture&& other) noexcept
{
	*this = std::move(other);
}

mvk::SurfaceTexture& mvk::SurfaceTexture::operator=(SurfaceTexture&& other) noexcept
{
    if (this != &other)
    {
        Destroy();
        m_image = other.m_image;
        m_imageView = other.m_imageView;
        m_extent = other.m_extent;
        m_format = other.m_format;
        m_samples = other.m_samples;
        m_currentLayout = other.m_currentLayout;
        other.m_image = nullptr;
        other.m_imageView = nullptr;
        other.m_currentLayout = vk::ImageLayout::eUndefined;
    }
    return *this;
}

void mvk::SurfaceTexture::Destroy()
{
    if (m_imageView)
    {
        VulkanCore::Instance()->GetDevice().destroyImageView(m_imageView);
        m_imageView = nullptr;
    }
    m_currentLayout = vk::ImageLayout::eUndefined;
}

void mvk::SurfaceTexture::TransitionLayout(vk::CommandBuffer cmdBuffer, vk::ImageLayout newLayout)
{
    if (m_currentLayout == newLayout) return;

    const auto transitionInfo = DetermineLayoutTransition(m_currentLayout, newLayout);
    const auto aspectMask = DetermineAspectMask();

    TransitionLayout(cmdBuffer, m_currentLayout, newLayout,
        transitionInfo.srcStage, transitionInfo.dstStage, aspectMask,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
}

void mvk::SurfaceTexture::TransitionLayout(
    vk::CommandBuffer cmdBuffer,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    vk::PipelineStageFlags srcStage,
    vk::PipelineStageFlags dstStage,
    vk::ImageAspectFlags aspectMask,
    uint32_t srcQueueFamilyIndex,
    uint32_t dstQueueFamilyIndex)
{
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

void mvk::SurfaceTexture::GenerateMipmaps(vk::CommandBuffer cmdBuffer)
{
    if (m_mipLevels <= 1) return;

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

vk::ImageAspectFlags mvk::SurfaceTexture::DetermineAspectMask() const
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

mvk::SurfaceTexture::LayoutTransitionInfo mvk::SurfaceTexture::DetermineLayoutTransition(
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
    else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR)
    {
        info.srcStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        info.dstStage = vk::PipelineStageFlagBits::eBottomOfPipe;
        info.srcAccess = vk::AccessFlagBits::eColorAttachmentWrite;
        info.dstAccess = vk::AccessFlagBits::eNone;
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

bool mvk::SurfaceTexture::IsDepthFormat(vk::Format format) const
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

bool mvk::SurfaceTexture::IsStencilFormat(vk::Format format) const
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

bool mvk::SurfaceTexture::IsDepthStencilFormat(vk::Format format) const
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