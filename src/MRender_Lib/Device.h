#pragma once

#include "VulkanDef.h"

#include "HeaderPre.h"

namespace mvk
{
	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphics;
		std::optional<uint32_t> present;
		constexpr bool IsComplete() const
		{
			return graphics.has_value() && present.has_value();
		}
		constexpr std::vector<uint32_t> Get() const
		{
			if (!IsComplete()) return {};

			auto i1 = graphics.value();
			auto i2 = present.value();
			if (i1 == i2)
				return { i1 };
			return { i1, i2 };
		}
		constexpr void reset()
		{
			graphics.reset();
			present.reset();
		}
	};

	class __MY_EXT_CLASS__ Device
	{
	public:
		Device(HWND window, const std::tuple<vk::Instance, vk::DebugUtilsMessengerEXT>& instance);
		~Device();

		void Initialize(HWND window);
		void Destroy();

	private:
		void CreateSurface(HWND window);
		void InitPhysicalDevice();
		void CreateDeviceAndQueue();

		QueueFamilyIndices FindQueueFamilies(const vk::PhysicalDevice& device) const;
		bool CheckDeviceExtensionSupport(const vk::PhysicalDevice& device) const;
		bool CheckSwapChainAdequate(const vk::PhysicalDevice& device) const;
		bool IsDeviceSuitable(const vk::PhysicalDevice& device) const;

	public:
		constexpr const vk::SurfaceKHR& GetSurface() const { return m_surface; }

		constexpr const vk::PhysicalDevice& GetPhysical() const { return m_physical; }
		constexpr const vk::Device& GetLogical() const { return m_logical; }

		constexpr const QueueFamilyIndices& GetQueueFamilyIndices() const { return m_queueFamilyIndices; }
		constexpr const vk::Queue& GetPresentQueue() const { return m_presentQueue; }
		constexpr const vk::Queue& GetQueue() const { return m_queue; }
		constexpr bool IsOneQueueFamily() const { return m_queueFamilyIndices.graphics == m_queueFamilyIndices.present; }

	private:
		vk::Instance m_instance;
		vk::DebugUtilsMessengerEXT m_debugMessenger;
		vk::SurfaceKHR m_surface;

		vk::PhysicalDevice m_physical;
		vk::Device m_logical;

		QueueFamilyIndices m_queueFamilyIndices;
		vk::Queue m_presentQueue;
		vk::Queue m_queue;
	};
}

#include "HeaderPost.h"

