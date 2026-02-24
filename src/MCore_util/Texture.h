#pragma once

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
    class __MY_EXT_CLASS__ Texture
    {
    public:
        Texture() = default;
        virtual ~Texture();
        Texture(const Texture&) = delete;
        Texture(Texture&& other) noexcept;
        Texture& operator=(const Texture&) = delete;
        Texture& operator=(Texture&& other) noexcept;

        void Create(vk::Format format, vk::ImageUsageFlags usage,
            vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1,
            vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined,
            uint32_t mipLevels = 1, uint32_t arrayLayers = 1);
        void Create(const vk::Extent3D& extent, vk::Format format, vk::ImageUsageFlags usage,
            vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1,
            vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined,
            uint32_t mipLevels = 1, uint32_t arrayLayers = 1);
        void Destroy();

        void Resize(const vk::Extent3D& extent);

        void CopyFromHost(const void* src, uint32_t layerIndex = UINT32_MAX);
        void CopyToHost(void* dst, uint32_t layerIndex = UINT32_MAX) const;
        void CopyToHost(void* dst, uint32_t x, uint32_t y, uint32_t layerIndex = UINT32_MAX) const;
        void Upload(vk::CommandBuffer cmdBuffer, uint32_t layerIndex = UINT32_MAX) const;
        void Download(vk::CommandBuffer cmdBuffer, uint32_t layerIndex = UINT32_MAX) const;
        void Download(vk::CommandBuffer cmdBuffer, uint32_t x, uint32_t y, uint32_t layerIndex = UINT32_MAX) const;

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

        constexpr vk::Extent3D GetExtent() const { return m_extent; }
        constexpr vk::Format GetFormat() const { return m_format; }
        constexpr const vk::Image& GetImage() const { return m_image; }
        constexpr const vk::ImageView& GetView() const { return m_imageView; }
        constexpr vk::ImageLayout GetCurrentLayout() const { return m_currentLayout; }
        constexpr uint32_t GetMipLevels() const { return m_mipLevels; }
        constexpr uint32_t GetArrayLayers() const { return m_arrayLayers; }
        constexpr vk::SampleCountFlagBits GetSampleCount() const { return m_samples; }
        constexpr vk::DeviceSize GetByteSize() const { return m_byteSize; }
        constexpr bool IsValid() const { return m_isCreated && m_image; }

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
        vk::DeviceMemory m_imageMemory;
        vk::Buffer m_stagingBuffer;
        vk::DeviceMemory m_stagingMemory;

        vk::Extent3D m_extent = {};
        vk::Format m_format = vk::Format::eUndefined;
        vk::ImageUsageFlags m_usage = {};
        vk::DeviceSize m_byteSize = 0;
        vk::ImageLayout m_currentLayout = vk::ImageLayout::eUndefined;
        vk::SampleCountFlagBits m_samples = vk::SampleCountFlagBits::e1;
        uint32_t m_mipLevels = 1;
        uint32_t m_arrayLayers = 1;
        bool m_isCreated = false;
    };
}

#include "HeaderPost.h"