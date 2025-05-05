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
	};

	class __MY_EXT_CLASS__ Device
	{
	public:
		Device(const vk::Instance& instance, const vk::SurfaceKHR& surface);
		~Device();

		void Initialize(const vk::Instance& instance, const vk::SurfaceKHR& surface);

		constexpr const vk::PhysicalDevice& GetPhysicalDevice() const { return m_physicalDevice; }
		constexpr const vk::Queue& GetPresentQueue() const { return m_queue; }

		constexpr const vk::Device& GetDevice() const { return m_device; }
		constexpr const vk::Queue& GetQueue() const { return m_queue; }

		constexpr const QueueFamilyIndices& GetQueueFamilyIndices() const { return m_queueFamilyIndices; }
		constexpr bool IsOneQueueFamily() const { return m_queueFamilyIndices.graphics == m_queueFamilyIndices.present; }

	private:
		void InitPhysicalDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface);
		void CreateDeviceAndQueue(const vk::SurfaceKHR& surface);

		QueueFamilyIndices FindQueueFamilies(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& device) const;
		bool CheckDeviceExtensionSupport(const vk::PhysicalDevice& device) const;
		bool CheckSwapChainAdequate(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& device) const;
		bool IsDeviceSuitable(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& device) const;

		vk::PhysicalDevice m_physicalDevice;
		vk::Queue m_presentQueue;

		vk::Device m_device;
		vk::Queue m_queue;

		QueueFamilyIndices m_queueFamilyIndices;
	};
}

#include "HeaderPost.h"

