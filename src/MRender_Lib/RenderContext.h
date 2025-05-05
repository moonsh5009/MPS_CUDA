#pragma once

#include "Device.h"

#include "HeaderPre.h"

namespace mvk
{
	class __MY_EXT_CLASS__ RenderContext
	{
	public:
		RenderContext(const std::shared_ptr<mvk::Device>& pDevice);
		~RenderContext();

		void Initialize();
		void Destroy();

	private:
		void CreateSwapChain();
		void GenerateImages();
		void CreateRenderPass();
		void CreateFrameBuffers();
		void CreateCommandPool();
		void CreateCommandBuffers();
		void CreateSyncObjects();

		void CleanupSwapChain();
		vk::Result SwapBuffer();
		void RecordCommandBuffer();

		vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const;
		vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const;
		vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width = 1, uint32_t height = 1) const;

	public:
		void DrawFrame();

		constexpr const vk::Image& GetCurrentImage() const { return m_images[m_imageIndex]; }
		constexpr const vk::ImageView& GetCurrentImageView() const { return m_imageViews[m_imageIndex]; }
		constexpr const vk::Framebuffer& GetCurrentFrameBuffer() const { return m_frameBuffers[m_imageIndex]; }

	private:
		vk::SwapchainKHR m_swapChain;
		vk::Format m_imageFormat;
		vk::Extent2D m_extent;
		std::vector<vk::Image> m_images;
		std::vector<vk::ImageView> m_imageViews;
		std::vector<vk::Framebuffer> m_frameBuffers;
		uint32_t m_imageIndex;

		vk::RenderPass m_renderPass;
		vk::CommandPool m_commandPool;
		std::array<vk::CommandBuffer, MAX_FRAMES_IN_FLIGHT> m_commandBuffers;
		std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> m_imageAvailableSemaphores;
		std::array<vk::Semaphore, MAX_FRAMES_IN_FLIGHT> m_renderFinishedSemaphores;
		std::array<vk::Fence, MAX_FRAMES_IN_FLIGHT> m_inFlightFences;
		uint32_t m_bufferIndex;

		bool m_framebufferResized;
		std::weak_ptr<mvk::Device> m_pDevice;
	};
}

#include "HeaderPost.h"

