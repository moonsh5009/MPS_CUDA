#pragma once

#include "Device.h"
#include "RenderContext.h"

#include "HeaderPre.h"

namespace mvk
{
	class __MY_EXT_CLASS__ Core
	{
	public:
		Core(HWND window);
		~Core();

		void Initialize(HWND window);
		void Destroy();

	private:
		std::tuple<vk::Instance, vk::DebugUtilsMessengerEXT> GenerateInstance();
		bool CheckValidationLayerSupport() const;
		std::vector<const char*> GetRequiredExtensions() const;

	public:
		static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
			vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			vk::DebugUtilsMessageTypeFlagsEXT messageType,
			const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData)
		{
			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
			return VK_FALSE;
		}

		constexpr const std::shared_ptr<RenderContext>& GetRenderContext() const { return m_pRenderContext; }

	private:
		std::shared_ptr<Device> m_pDevice;
		std::shared_ptr<RenderContext> m_pRenderContext;
	};
}

#include "HeaderPost.h"

