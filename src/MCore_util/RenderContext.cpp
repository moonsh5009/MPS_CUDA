#include "stdafx.h"
#include "RenderContext.h"

#include "VulkanCore.h"

mvk::RenderContext::RenderContext()
    : m_imageIndex{ 0 }
    ,  m_inFlightIndex{ 0 }
{
}

mvk::RenderContext::~RenderContext()
{
    Destroy();
}

void mvk::RenderContext::Initialize(HWND window)
{
	const auto pCore = VulkanCore::Instance();

    m_surface = pCore->CreateSurface(window);
    if (!m_surface)
    {
        throw std::runtime_error("Can't Create Vulkan Win32 Surface");
    }

    m_pCommander = std::make_shared<Commander>(weak_from_this());
    m_pCommander->Initialize();

    m_pCommandQueue = std::make_shared<CommandQueue>(m_pCommander);

    CreateSwapChain();
    CreateTextures();
    CreateSyncObjects();
    CreateBarriers();

    m_imageIndex = 0;
    m_inFlightIndex = 0;
}

void mvk::RenderContext::Destroy()
{
    const auto pCore = VulkanCore::Instance();

    m_surfaceTextures.clear();

    pCore->GetDevice().destroySwapchainKHR(m_swapChain);
    m_swapChain = nullptr;

    m_pCommander->Destroy();

    pCore->GetInstance().destroySurfaceKHR(m_surface);
    m_surface = nullptr;
}

void mvk::RenderContext::CreateSwapChain()
{
    const auto pCore = VulkanCore::Instance();

    const auto formats = pCore->GetPhysicalDevice().getSurfaceFormatsKHR(m_surface);
    const auto presentModes = pCore->GetPhysicalDevice().getSurfacePresentModesKHR(m_surface);
    const auto capabilities = pCore->GetPhysicalDevice().getSurfaceCapabilitiesKHR(m_surface);

    const auto surfaceFormat = ChooseSwapSurfaceFormat(formats);
    const auto presentMode = ChooseSwapPresentMode(presentModes);
    const auto extent = ChooseSwapExtent(capabilities);
    if (extent.width == 0 || extent.height == 0)
    {
        throw std::runtime_error("SwapChain extent is invalid");
	}

    auto imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    if (capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferSrc)
    {
        imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    }
    if (capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)
    {
        imageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    }

    vk::SwapchainCreateInfoKHR createInfo(
        {}, m_surface, imageCount,
        surfaceFormat.format, surfaceFormat.colorSpace,
        extent, 1, imageUsage
    );

    const auto indices = pCore->GetQueueFamilyIndices().GetSwapChainQueueIndices();
    if (indices.size() == 1)
    {
        createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
    }
    else
    {
        createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        createInfo.setQueueFamilyIndices(indices);
    }

    if (m_swapChain)
    {
        createInfo.oldSwapchain = m_swapChain;
        auto newSwapChain = pCore->GetDevice().createSwapchainKHR(createInfo);
        pCore->GetDevice().destroySwapchainKHR(m_swapChain);

        m_swapChain = newSwapChain;
    }
    else
    {
        m_swapChain = pCore->GetDevice().createSwapchainKHR(createInfo);
    }
}

void mvk::RenderContext::CreateTextures()
{
    const auto pCore = VulkanCore::Instance();

    const auto images = pCore->GetDevice().getSwapchainImagesKHR(m_swapChain);
    m_surfaceTextures.reserve(images.size());

    const auto formats = pCore->GetPhysicalDevice().getSurfaceFormatsKHR(m_surface);
    const auto capabilities = pCore->GetPhysicalDevice().getSurfaceCapabilitiesKHR(m_surface);
    const auto surfaceFormat = ChooseSwapSurfaceFormat(formats);
    const auto extent = ChooseSwapExtent(capabilities);

    for (const auto& image : images)
    {
        m_surfaceTextures.emplace_back(std::make_shared<SurfaceTexture>(image, extent, surfaceFormat.format));
    }
    m_imageSize = extent;
    m_imageFormat = surfaceFormat.format;
}

void mvk::RenderContext::CreateSyncObjects()
{
    const auto pCore = VulkanCore::Instance();

    vk::FenceCreateInfo fenceInfo{};
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_renderReadySemaphore[i] = GetCommander()->CreateSemaphore();
        m_renderFinishedFuture[i].Reset();
    }
}

void mvk::RenderContext::CreateBarriers()
{
    vk::CommandBufferBeginInfo info{};
    m_imageReadyCommand.reserve(m_surfaceTextures.size());
    m_imageFinishedCommand.reserve(m_surfaceTextures.size());
    for (size_t i = 0; i < m_surfaceTextures.size(); i++)
    {
        auto readyCommandBufferID = GetCommander()->CreateCommandBuffer(mvk::QueueType::GRAPHIC, mvk::CommandType::REUSABLE);
        const auto readyCommandBuffer = GetCommander()->Get(readyCommandBufferID);
        readyCommandBuffer.begin(info);
        m_surfaceTextures[i]->TransitionLayout(readyCommandBuffer, vk::ImageLayout::eColorAttachmentOptimal);
        readyCommandBuffer.end();

        auto finishedCommandBufferID = GetCommander()->CreateCommandBuffer(mvk::QueueType::GRAPHIC, mvk::CommandType::REUSABLE);
        const auto finishedCommandBuffer = GetCommander()->Get(finishedCommandBufferID);
        finishedCommandBuffer.begin(info);
        m_surfaceTextures[i]->TransitionLayout(finishedCommandBuffer, vk::ImageLayout::ePresentSrcKHR);
        finishedCommandBuffer.end();

        m_imageReadyCommand.emplace_back(std::move(readyCommandBufferID));
        m_imageFinishedCommand.emplace_back(std::move(finishedCommandBufferID));
    }
}

void mvk::RenderContext::RecreateSwapChain()
{
    const auto pCore = VulkanCore::Instance();

    const auto& pDevice = pCore->GetDevice();
    pCore->GetDevice().waitIdle();

    for (auto& future : m_renderFinishedFuture)
    {
        future.Wait();
    }

    m_surfaceTextures.clear();
    m_imageReadyCommand.clear();
    m_imageFinishedCommand.clear();

    CreateSwapChain();
    CreateTextures();
    CreateBarriers();
}

vk::Result mvk::RenderContext::SwapBuffer()
{
    try
    {
        const auto pCore = VulkanCore::Instance();

        m_renderFinishedFuture[m_inFlightIndex].Wait();

        const auto renderReadySemaphore = GetCommander()->Get(m_renderReadySemaphore[m_inFlightIndex]);
        const auto result = pCore->GetDevice().acquireNextImageKHR(
            m_swapChain,
            UINT64_MAX,
            renderReadySemaphore
        );
        m_imageIndex = result.value;
        return result.result;
    }
    catch (vk::OutOfDateKHRError&)
    {
        return vk::Result::eErrorOutOfDateKHR;
    }
    catch (vk::SystemError& err)
    {
        mcore::Logger::Error("SwapBuffer failed", err.what());
        mcore::Logger::Print();
        throw;
    }
}

vk::SurfaceFormatKHR mvk::RenderContext::ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const
{
    constexpr std::array preferredFormats = {
        vk::SurfaceFormatKHR{ vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear },
        vk::SurfaceFormatKHR{ vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear },
        vk::SurfaceFormatKHR{ vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear },
        vk::SurfaceFormatKHR{ vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear }
    };

    for (const auto& preferred : preferredFormats)
    {
        for (const auto& available : availableFormats)
        {
            if (available.format == preferred.format &&
                available.colorSpace == preferred.colorSpace)
            {
                return available;
            }
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR mvk::RenderContext::ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const
{
    constexpr std::array preferredModes = {
        vk::PresentModeKHR::eMailbox,
        vk::PresentModeKHR::eFifoRelaxed,
        vk::PresentModeKHR::eFifo,
        vk::PresentModeKHR::eImmediate
    };

    for (const auto& preferred : preferredModes)
    {
        for (const auto& available : availablePresentModes)
        {
            if (available == preferred)
            {
                return available;
            }
        }
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

vk::Extent2D mvk::RenderContext::GetSurfaceExtent() const
{
    const auto pCore = VulkanCore::Instance();
    const auto capabilities = pCore->GetPhysicalDevice().getSurfaceCapabilitiesKHR(m_surface);
	return ChooseSwapExtent(capabilities);
}

std::shared_ptr<mvk::SurfaceTexture> mvk::RenderContext::GetNextSurfaceTexture()
{
    const auto result = SwapBuffer();
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        RecreateSwapChain();
        return {};
    }
    else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    m_pCommandQueue->AddSemaphore(m_renderReadySemaphore[m_inFlightIndex], vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    m_pCommandQueue->AddCommand(m_imageReadyCommand[m_imageIndex]);
    m_pCommandQueue->FlushAsync(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    return m_surfaceTextures[m_imageIndex];
}

void mvk::RenderContext::Present()
{
    const auto pCore = VulkanCore::Instance();

    m_pCommandQueue->AddCommand(m_imageFinishedCommand[m_imageIndex]);
    //m_pCommandQueue->FlushAsync(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    m_renderFinishedFuture[m_inFlightIndex] = m_pCommandQueue->Flush(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    
    auto semaphores = m_pCommandQueue->GetWaitSemaphores();
    std::vector<vk::Semaphore> waitSemaphores;
    waitSemaphores.reserve(semaphores.size());
    std::ranges::transform(semaphores, std::back_inserter(waitSemaphores), [&](const auto& semaphoreInfo)
    {
        const auto& [semaphoreID, _] = semaphoreInfo;
        return GetCommander()->Get(semaphoreID);
    });

    vk::Result result;
    vk::PresentInfoKHR presentInfo{
        waitSemaphores,
        m_swapChain,
        m_imageIndex,
        result,
    };

    result = pCore->GetPresentQueue().presentKHR(presentInfo);

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
    {
        RecreateSwapChain();
    }
    else if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    m_pCommandQueue->Reset();
    m_inFlightIndex = (m_inFlightIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

bool mvk::RenderContext::IsMultisamplingSupported(vk::SampleCountFlagBits samples) const
{
    const auto props = VulkanCore::Instance()->GetPhysicalDevice().getProperties();
    return (props.limits.framebufferColorSampleCounts & samples) &&
        (props.limits.framebufferDepthSampleCounts & samples);
}