#include "stdafx.h"
#include "Surface.h"

mvk::Surface::Surface(vk::SurfaceKHR&& surface, const std::shared_ptr<mvk::Device>& pDevice) :
    m_surface{ std::move(surface) },
    m_pDevice{ pDevice }
{
    Initialize();
}

mvk::Surface::~Surface()
{}

void mvk::Surface::Initialize()
{
    CreateSwapChain();
    GenerateImageView();
}

void mvk::Surface::CreateSwapChain()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    const auto formats = pDevice->GetPhysicalDevice().getSurfaceFormatsKHR(m_surface);
    const auto presentModes = pDevice->GetPhysicalDevice().getSurfacePresentModesKHR(m_surface);
    const auto capabilities = pDevice->GetPhysicalDevice().getSurfaceCapabilitiesKHR(m_surface);

    const auto surfaceFormat = ChooseSwapSurfaceFormat(formats);
    const auto presentMode = ChooseSwapPresentMode(presentModes);
    const auto extent = ChooseSwapExtent(capabilities);

    auto imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo(
        {}, m_surface, imageCount,
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

    m_swapChain = pDevice->GetDevice().createSwapchainKHR(createInfo);

    m_imageFormat = surfaceFormat.format;
    m_extent = extent;
    m_swapChainImages = pDevice->GetDevice().getSwapchainImagesKHR(m_swapChain);
}

void mvk::Surface::GenerateImageView()
{
    const auto pDevice = m_pDevice.lock();
    if (!pDevice)
    {
        throw std::runtime_error("lost device");
    }

    m_swapChainImageViews.clear();
    m_swapChainImageViews.reserve(m_swapChainImages.size());

    for (const auto& image : m_swapChainImages)
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
        m_swapChainImageViews.emplace_back(pDevice->GetDevice().createImageView(viewCreateInfo));
    }
}

vk::SurfaceFormatKHR mvk::Surface::ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return availableFormat;
    }
    return availableFormats[0];
}

vk::PresentModeKHR mvk::Surface::ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            return availablePresentMode;
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D mvk::Surface::ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) const
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    else
    {
        return {
            std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }
}