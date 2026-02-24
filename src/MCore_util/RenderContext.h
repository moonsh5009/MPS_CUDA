#pragma once

#include "SurfaceTexture.h"
#include "Commander.h"

#include "HeaderPre.h"

namespace mvk
{
	class __MY_EXT_CLASS__ RenderContext : public std::enable_shared_from_this<RenderContext>
	{
	public:
		RenderContext();
		~RenderContext();
		RenderContext(const RenderContext&) = delete;
		RenderContext(RenderContext&&) = default;
		RenderContext& operator=(const RenderContext&) = delete;
		RenderContext& operator=(RenderContext&&) = default;

		void Initialize(HWND window);
		void Destroy();

	private:
		void CreateSwapChain();
		void CreateTextures();
		void CreateSyncObjects();
		void CreateBarriers();

		void RecreateSwapChain();
		vk::Result SwapBuffer();

		vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const;
		vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const;
		vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width = 0, uint32_t height = 0) const;

	public:
		void Resize() { RecreateSwapChain(); }

		vk::Extent2D GetSurfaceExtent() const;
		std::shared_ptr<SurfaceTexture> GetNextSurfaceTexture();
		void Present();
		bool IsMultisamplingSupported(vk::SampleCountFlagBits samples) const;

		constexpr uint32_t GetMaxFramesinFlight() const { return MAX_FRAMES_IN_FLIGHT; }
		uint32_t GetInFlightIndex() const { return m_inFlightIndex; }
		vk::Extent2D GetImageSize() const { return m_imageSize; }
		vk::Viewport GetViewport() const { return { 0.f, 0.f, static_cast<float>(m_imageSize.width), static_cast<float>(m_imageSize.height), 0.f, 1.f }; }
		vk::Rect2D GetRenderArea() const { return { { 0, 0 }, m_imageSize }; }
		vk::Format GetFormat() const { return m_imageFormat; }

		const std::shared_ptr<Commander>& GetCommander() const { return m_pCommander; }
		const std::shared_ptr<CommandQueue>& GetCommandQueue() const { return m_pCommandQueue; }

	private:
		vk::SurfaceKHR m_surface;
		vk::SwapchainKHR m_swapChain;

		vk::Extent2D m_imageSize;
		vk::Format m_imageFormat;

		std::vector<std::shared_ptr<SurfaceTexture>> m_surfaceTextures;
		std::vector<CommandID> m_imageReadyCommand;
		std::vector<CommandID> m_imageFinishedCommand;
		uint32_t m_imageIndex;

		std::array<SemaphoreID, MAX_FRAMES_IN_FLIGHT> m_renderReadySemaphore;
		std::array<Future, MAX_FRAMES_IN_FLIGHT> m_renderFinishedFuture;
		uint32_t m_inFlightIndex;

		std::shared_ptr<Commander> m_pCommander;
		std::shared_ptr<CommandQueue> m_pCommandQueue;
	};
}

#include "HeaderPost.h"

