#include "stdafx.h"
#include "RenderContext.h"

mvk::RenderContext::RenderContext(const std::shared_ptr<mvk::Device>& pDevice) :
    m_pDevice{ pDevice },
    m_imageIndex{ 0 },
    m_bufferIndex{ 0 },
    m_framebufferResized{ false }
{
    Initialize();
}

mvk::RenderContext::~RenderContext()
{
    Destroy();
}

void mvk::RenderContext::Initialize()
{
    CreateSwapChain();
    GenerateImages();
    CreateRenderPass();
    CreateFrameBuffers();
    CreateCommandPool();
    CreateCommandBuffers();
    CreateSyncObjects();

    m_imageIndex = 0;
    m_bufferIndex = 0;
    m_framebufferResized = false;
}

void mvk::RenderContext::Destroy()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    CleanupSwapChain();

    pDevice->GetLogical().destroyRenderPass(m_renderPass);
    m_renderPass = nullptr;

    pDevice->GetLogical().destroyCommandPool(m_commandPool);
    m_commandPool = nullptr;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        pDevice->GetLogical().destroySemaphore(m_renderFinishedSemaphores[i]);
        m_renderFinishedSemaphores[i] = nullptr;

        pDevice->GetLogical().destroySemaphore(m_imageAvailableSemaphores[i]);
        m_imageAvailableSemaphores[i] = nullptr;

        pDevice->GetLogical().destroyFence(m_inFlightFences[i]);
        m_inFlightFences[i] = nullptr;
    }
}

void mvk::RenderContext::CreateSwapChain()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    const auto formats = pDevice->GetPhysical().getSurfaceFormatsKHR(pDevice->GetSurface());
    const auto presentModes = pDevice->GetPhysical().getSurfacePresentModesKHR(pDevice->GetSurface());
    const auto capabilities = pDevice->GetPhysical().getSurfaceCapabilitiesKHR(pDevice->GetSurface());

    const auto surfaceFormat = ChooseSwapSurfaceFormat(formats);
    const auto presentMode = ChooseSwapPresentMode(presentModes);
    const auto extent = ChooseSwapExtent(capabilities);

    auto imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo(
        {}, pDevice->GetSurface(), imageCount,
        surfaceFormat.format, surfaceFormat.colorSpace,
        extent, 1, vk::ImageUsageFlagBits::eColorAttachment
    );

    if (pDevice->IsOneQueueFamily())
    {
        createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
    }
    else
    {
        const auto indices = pDevice->GetQueueFamilyIndices().Get();
        createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        createInfo.setQueueFamilyIndices(indices);
    }

    m_imageFormat = surfaceFormat.format;
    m_extent = extent;
    m_swapChain = pDevice->GetLogical().createSwapchainKHR(createInfo);
}

void mvk::RenderContext::GenerateImages()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    m_images = pDevice->GetLogical().getSwapchainImagesKHR(m_swapChain);
    m_imageViews.clear();
    m_imageViews.reserve(m_images.size());

    for (const auto& image : m_images)
    {
        vk::ImageViewCreateInfo viewCreateInfo(
            {},
            image,
            vk::ImageViewType::e2D,
            m_imageFormat,
            vk::ComponentMapping(),
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor,
                0, 1, 0, 1)
        );
        m_imageViews.emplace_back(pDevice->GetLogical().createImageView(viewCreateInfo));
    }
}

void mvk::RenderContext::CreateRenderPass()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    vk::AttachmentDescription colorAttachment{ {},
        m_imageFormat,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::ePresentSrcKHR,
    };
    vk::AttachmentReference colorAttachmentRef{
        0,
        vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass{ {},
        vk::PipelineBindPoint::eGraphics,
        {},
        colorAttachmentRef,
    };
    vk::SubpassDependency dependency{
        VK_SUBPASS_EXTERNAL,
        0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::AccessFlagBits::eNone,
        vk::AccessFlagBits::eColorAttachmentWrite,
    };
    vk::RenderPassCreateInfo createInfo{ {},
        colorAttachment,
        subpass,
        dependency,
    };

    m_renderPass = pDevice->GetLogical().createRenderPass(createInfo);
    if (!m_renderPass)
    {
        throw std::runtime_error("failed to create render pass!");
    }
}

void mvk::RenderContext::CreateFrameBuffers()
{
    assert(m_frameBuffers.empty());
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    m_frameBuffers.reserve(m_imageViews.size());
    for (const auto& view : m_imageViews)
    {
        vk::FramebufferCreateInfo createInfo{ {},
            m_renderPass,
            view,
            m_extent.width,
            m_extent.height,
            1,
        };

        m_frameBuffers.emplace_back(pDevice->GetLogical().createFramebuffer(createInfo));
        if (!m_frameBuffers.back())
        {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void mvk::RenderContext::CreateCommandPool()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    vk::CommandPoolCreateInfo createInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        pDevice->GetQueueFamilyIndices().graphics.value(),
    };

    m_commandPool = pDevice->GetLogical().createCommandPool(createInfo);
    if (!m_commandPool)
    {
        throw std::runtime_error("failed to create command pool!");
    }
}

void mvk::RenderContext::CreateCommandBuffers()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    vk::CommandBufferAllocateInfo allocInfo{
        m_commandPool,
        vk::CommandBufferLevel::ePrimary,
        static_cast<uint32_t>(m_commandBuffers.size()),
    };
    auto buffers = pDevice->GetLogical().allocateCommandBuffers(allocInfo);
    if (buffers.size() != m_commandBuffers.size())
    {
        throw std::runtime_error("failed to allocate command buffers!");
    }

    std::copy(buffers.begin(), buffers.end(), m_commandBuffers.begin());
}

void mvk::RenderContext::CreateSyncObjects()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{vk::FenceCreateFlagBits::eSignaled};
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_imageAvailableSemaphores[i] = pDevice->GetLogical().createSemaphore(semaphoreInfo);
        m_renderFinishedSemaphores[i] = pDevice->GetLogical().createSemaphore(semaphoreInfo);
        m_inFlightFences[i] = pDevice->GetLogical().createFence(fenceInfo);
        if (!m_imageAvailableSemaphores[i] || !m_renderFinishedSemaphores[i] || !m_inFlightFences[i])
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void mvk::RenderContext::CleanupSwapChain()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    for (auto frameBuffer : m_frameBuffers)
    {
        pDevice->GetLogical().destroyFramebuffer(frameBuffer);
    }
    m_frameBuffers.clear();

    for (auto imageView : m_imageViews)
    {
        pDevice->GetLogical().destroyImageView(imageView);
    }
    m_imageViews.clear();

    pDevice->GetLogical().destroySwapchainKHR(m_swapChain);
    m_swapChain = nullptr;
}

vk::Result mvk::RenderContext::SwapBuffer()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    if (pDevice->GetLogical().waitForFences(m_inFlightFences[m_bufferIndex], vk::True, UINT64_MAX) != vk::Result::eSuccess)
    {
        throw std::runtime_error("waitForFences");
    }

    const auto result = pDevice->GetLogical().acquireNextImageKHR(m_swapChain, UINT64_MAX, m_imageAvailableSemaphores[m_bufferIndex]);
    m_imageIndex = result.value;

    return result.result;
}

void mvk::RenderContext::RecordCommandBuffer()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    const auto& commandBuffer = m_commandBuffers[m_bufferIndex];

    vk::CommandBufferBeginInfo beginInfo{};
    vk::ClearValue clearColor{ vk::ClearColorValue{ 1.0f, 1.0f, 0.0f, 1.0f } };

    vk::RenderPassBeginInfo renderPassInfo{
        m_renderPass,
        m_frameBuffers[m_imageIndex],
        { { 0, 0 }, m_extent },
        clearColor
    };

    commandBuffer.begin(beginInfo);
    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    /*
    vk::Viewport viewport{
        0.0f,
        0.0f,
        static_cast<float>(m_extent.width),
        static_cast<float>(m_extent.height),
        0.0f,
        1.0f,
    };
    vk::Rect2D scissor{ { 0, 0 }, m_extent };
    commandBuffer.bindPipeline(pipelineBindPoint, vk::PipelineBindPoint::eGraphics);
    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);
    commandBuffer.draw(3, 1, 0, 0);*/

    commandBuffer.endRenderPass();
    commandBuffer.end();
}

void mvk::RenderContext::DrawFrame()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    {
        const auto result = SwapBuffer();
        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            //recreateSwapChain();
            return;
        }
        else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
    }

    const auto& commandBuffer = m_commandBuffers[m_bufferIndex];

    pDevice->GetLogical().resetFences(m_inFlightFences[m_bufferIndex]);
    
    commandBuffer.reset();
    RecordCommandBuffer();

    vk::PipelineStageFlags waitDstStageMask{ vk::PipelineStageFlagBits::eColorAttachmentOutput };
    vk::SubmitInfo submitInfo{
        m_imageAvailableSemaphores[m_bufferIndex],
        waitDstStageMask,
        commandBuffer,
        m_renderFinishedSemaphores[m_bufferIndex],
    };
    pDevice->GetQueue().submit(submitInfo, m_inFlightFences[m_bufferIndex]);

    vk::Result result;
    vk::PresentInfoKHR presentInfo{
        m_renderFinishedSemaphores[m_bufferIndex],
        m_swapChain,
        m_imageIndex,
        result,
    };
    if (pDevice->GetPresentQueue().presentKHR(presentInfo) != vk::Result::eSuccess)
    {
        throw std::runtime_error("presentKHR");
    }
    
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || m_framebufferResized)
    {
        m_framebufferResized = false;
        //recreateSwapChain();
    }
    else if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    m_bufferIndex = (m_bufferIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    pDevice->GetLogical().waitIdle();
}

vk::SurfaceFormatKHR mvk::RenderContext::ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return availableFormat;
    }
    return availableFormats[0];
}

vk::PresentModeKHR mvk::RenderContext::ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            return availablePresentMode;
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D mvk::RenderContext::ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) const
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    return {
        std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}