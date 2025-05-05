#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan/vulkan.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace mvk
{
#ifdef _DEBUG
	constexpr bool ENABLE_VALIDATION_LAYERS = true;
#else
	constexpr bool ENABLE_VALIDATION_LAYERS = false;
#endif

	constexpr std::array<const char*, 1> VALIDATION_LAYERS = {
		"VK_LAYER_KHRONOS_validation"
	};
	constexpr std::array<const char*, 1> DEVICE_EXTENSIONS = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};
}