#pragma once

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ SurfaceTexture
    {
    public:
        SurfaceTexture(
            vk::Image image,
            const vk::Extent2D& extent,
            vk::Format format,
            vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);
        ~SurfaceTexture();
        SurfaceTexture(const SurfaceTexture&) = delete;
        SurfaceTexture(SurfaceTexture&& other) noexcept;
        SurfaceTexture& operator=(const SurfaceTexture&) = delete;
        SurfaceTexture& operator=(SurfaceTexture&& other) noexcept;

        void Destroy();

        void CopyFromHost(const void* src, uint32_t layerIndex = UINT32_MAX);
        void CopyToHost(void* dst, uint32_t layerIndex = UINT32_MAX) const;
        void Upload(vk::CommandBuffer cmdBuffer, uint32_t layerIndex = UINT32_MAX);
        void Download(vk::CommandBuffer cmdBuffer, uint32_t layerIndex = UINT32_MAX);

        void TransitionLayout(vk::CommandBuffer cmdBuffer, vk::ImageLayout newLayout);
        void TransitionLayout(vk::CommandBuffer cmdBuffer,
            vk::ImageLayout oldLayout,
            vk::ImageLayout newLayout,
            vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eAllCommands,
            vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor,
            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

        void GenerateMipmaps(vk::CommandBuffer cmdBuffer);

        constexpr vk::Extent2D GetExtent() const { return m_extent; }
        constexpr vk::Format GetFormat() const { return m_format; }
        constexpr const vk::Image& GetImage() const { return m_image; }
        constexpr const vk::ImageView& GetView() const { return m_imageView; }
        constexpr vk::ImageLayout GetCurrentLayout() const { return m_currentLayout; }
        constexpr uint32_t GetMipLevels() const { return m_mipLevels; }
        constexpr uint32_t GetArrayLayers() const { return m_arrayLayers; }
        constexpr vk::SampleCountFlagBits GetSampleCount() const { return m_samples; }

    private:
        struct LayoutTransitionInfo
        {
            vk::PipelineStageFlags srcStage;
            vk::PipelineStageFlags dstStage;
            vk::AccessFlags srcAccess;
            vk::AccessFlags dstAccess;
        };

        vk::ImageAspectFlags DetermineAspectMask() const;
        LayoutTransitionInfo DetermineLayoutTransition(vk::ImageLayout oldLayout, vk::ImageLayout newLayout) const;
        bool IsDepthFormat(vk::Format format) const;
        bool IsStencilFormat(vk::Format format) const;
        bool IsDepthStencilFormat(vk::Format format) const;
        vk::DeviceSize CalculateLayerSize() const;

    protected:
        vk::Image m_image;
        vk::ImageView m_imageView;
        vk::Extent2D m_extent = {};
        vk::Format m_format = vk::Format::eUndefined;
        vk::ImageLayout m_currentLayout = vk::ImageLayout::eUndefined;
        vk::SampleCountFlagBits m_samples = vk::SampleCountFlagBits::e1;
        uint32_t m_mipLevels = 1;
        uint32_t m_arrayLayers = 1;
    };
}

#include "HeaderPost.h"