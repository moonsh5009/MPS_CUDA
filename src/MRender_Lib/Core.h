#pragma once

#include "Device.h"
#include "Surface.h"

#include "HeaderPre.h"

namespace mvk
{
	class __MY_EXT_CLASS__ Core
	{
	public:
		Core(HWND window);
		~Core();

		void Initialize(HWND window);
		void GenerateInstance();

		static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
			vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			vk::DebugUtilsMessageTypeFlagsEXT messageType,
			const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData)
		{
			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
			return VK_FALSE;
		}

	private:
		bool CheckValidationLayerSupport() const;
		std::vector<const char*> GetRequiredExtensions() const;

		vk::Instance m_instance;
		vk::DebugUtilsMessengerEXT debugMessenger;

		std::shared_ptr<Device> m_pDevice;
		std::shared_ptr<Surface> m_pSurface;
	};
}

#include "HeaderPost.h"

