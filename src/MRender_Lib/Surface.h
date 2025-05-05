#pragma once

#include "Device.h"

#include "HeaderPre.h"

namespace mvk
{
	class __MY_EXT_CLASS__ Surface
	{
	public:
		Surface(vk::SurfaceKHR&& surface, const std::shared_ptr<mvk::Device>& pDevice);
		~Surface();

		void Initialize();

	private:
		void CreateSwapChain();
		void GenerateImageView();

		vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const;
		vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const;
		vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width = 1, uint32_t height = 1) const;

	private:
		vk::SurfaceKHR m_surface;
		vk::SwapchainKHR m_swapChain;
		vk::Format m_imageFormat;
		vk::Extent2D m_extent;
		std::vector<vk::Image> m_swapChainImages;
		std::vector<vk::ImageView> m_swapChainImageViews;

		std::weak_ptr<mvk::Device> m_pDevice;
	};
}

#include "HeaderPost.h"

