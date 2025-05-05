#include "stdafx.h"
#include "Device.h"

#include <string>
#include <set>

mvk::Device::Device(const vk::Instance& instance, const vk::SurfaceKHR& surface)
{
	Initialize(instance, surface);
}

mvk::Device::~Device()
{}

void mvk::Device::Initialize(const vk::Instance& instance, const vk::SurfaceKHR& surface)
{
	InitPhysicalDevice(instance, surface);
	CreateDeviceAndQueue(surface);
}

void mvk::Device::InitPhysicalDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface)
{
	const auto physicalDevices = instance.enumeratePhysicalDevices();
	if (physicalDevices.empty())
		throw std::runtime_error("failed to find GPUs with Vulkan support!");

	for (const auto& physicalDevice : physicalDevices)
	{
		if (IsDeviceSuitable(surface, physicalDevice))
		{
			m_physicalDevice = physicalDevice;
			return;
		}
	}

	throw std::runtime_error("failed to find a suitable GPU!");
}

void mvk::Device::CreateDeviceAndQueue(const vk::SurfaceKHR& surface)
{
	auto queueFamilies = FindQueueFamilies(surface, m_physicalDevice);
	const auto queuePriority = 1.0f;
	const auto queueFamilyIndices = queueFamilies.Get();

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	queueCreateInfos.reserve(queueFamilyIndices.size());
	for (auto queueFamilyIndex : queueFamilyIndices)
	{
		queueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{ {}, queueFamilyIndex, 1, &queuePriority });
	}

	vk::DeviceCreateInfo deviceCreateInfo({},
		queueCreateInfos.size(),
		queueCreateInfos.data());

	if (ENABLE_VALIDATION_LAYERS)
	{
		deviceCreateInfo.setPEnabledLayerNames(VALIDATION_LAYERS);
	}
	deviceCreateInfo.setPEnabledExtensionNames(DEVICE_EXTENSIONS);

	m_device = m_physicalDevice.createDevice(deviceCreateInfo);
	m_queue = m_device.getQueue(*queueFamilies.graphics, 0);
	m_presentQueue = m_device.getQueue(*queueFamilies.present, 0);
	m_queueFamilyIndices = std::move(queueFamilies);
}

mvk::QueueFamilyIndices mvk::Device::FindQueueFamilies(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& device) const
{
	QueueFamilyIndices indices;

	const auto queueFamilyProperties = device.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
	{
		if (queueFamilyProperties[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))
			indices.graphics = i;
		if (device.getSurfaceSupportKHR(i, surface))
			indices.present = i;
		if (indices.IsComplete())
			return indices;
	}

	throw std::runtime_error("No Graphics Queue Family found!");
}

bool mvk::Device::CheckDeviceExtensionSupport(const vk::PhysicalDevice& device) const
{
	const auto availableExtensions = device.enumerateDeviceExtensionProperties();
	std::set<std::string_view> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
	for (const auto& extension : availableExtensions)
	{
		requiredExtensions.erase(extension.extensionName);
	}
	return requiredExtensions.empty();
}

bool mvk::Device::CheckSwapChainAdequate(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& device) const
{
	const auto formats = device.getSurfaceFormatsKHR(surface);
	const auto presentModes = device.getSurfacePresentModesKHR(surface);
	return !formats.empty() && !presentModes.empty();
}

bool mvk::Device::IsDeviceSuitable(const vk::SurfaceKHR& surface, const vk::PhysicalDevice& device) const
{
	return FindQueueFamilies(surface, device).IsComplete()
		&& CheckDeviceExtensionSupport(device)
		&& CheckSwapChainAdequate(surface, device);
}