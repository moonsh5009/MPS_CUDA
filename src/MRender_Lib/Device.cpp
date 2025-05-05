#include "stdafx.h"
#include "Device.h"

#include "Core.h"
#include <string>
#include <set>

mvk::Device::Device(HWND window, const std::tuple<vk::Instance, vk::DebugUtilsMessengerEXT>& instance) :
	m_instance{ std::get<0>(instance) },
	m_debugMessenger{ std::get<1>(instance) }
{
	Initialize(window);
}

mvk::Device::~Device()
{
	Destroy();
}

void mvk::Device::Initialize(HWND window)
{
	CreateSurface(window);
	InitPhysicalDevice();
	CreateDeviceAndQueue();
}

void mvk::Device::Destroy()
{
	m_logical.destroy();
	m_logical = nullptr;

	//m_instance.destroyDebugUtilsMessengerEXT(m_debugMessenger);
	//m_debugMessenger = nullptr;

	m_instance.destroySurfaceKHR(m_surface);
	m_surface = nullptr;

	m_instance.destroy();
	m_instance = nullptr;

	m_physical = nullptr;
	m_presentQueue = nullptr;
	m_queue = nullptr;
	m_queueFamilyIndices.reset();
}

void mvk::Device::CreateSurface(HWND window)
{
	vk::Win32SurfaceCreateInfoKHR createInfo({}, GetModuleHandle(nullptr), window);
	m_surface = m_instance.createWin32SurfaceKHR(createInfo);
	if (!m_surface)
	{
		throw std::runtime_error("Can't Create Vulkan Win32 Surface");
	}
}

void mvk::Device::InitPhysicalDevice()
{
	const auto physicalDevices = m_instance.enumeratePhysicalDevices();
	if (physicalDevices.empty())
		throw std::runtime_error("failed to find GPUs with Vulkan support!");

	for (const auto& physicalDevice : physicalDevices)
	{
		if (IsDeviceSuitable(physicalDevice))
		{
			m_physical = physicalDevice;
			return;
		}
	}

	throw std::runtime_error("failed to find a suitable GPU!");
}

void mvk::Device::CreateDeviceAndQueue()
{
	auto queueFamilies = FindQueueFamilies(m_physical);
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
		deviceCreateInfo.setPEnabledLayerNames(VALIDATION_LAYERS);
	deviceCreateInfo.setPEnabledExtensionNames(DEVICE_EXTENSIONS);

	m_logical = m_physical.createDevice(deviceCreateInfo);
	m_queue = m_logical.getQueue(*queueFamilies.graphics, 0);
	m_presentQueue = m_logical.getQueue(*queueFamilies.present, 0);
	m_queueFamilyIndices = std::move(queueFamilies);
}

mvk::QueueFamilyIndices mvk::Device::FindQueueFamilies(const vk::PhysicalDevice& device) const
{
	QueueFamilyIndices indices;

	const auto queueFamilyProperties = device.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
	{
		if (queueFamilyProperties[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))
			indices.graphics = i;
		if (device.getSurfaceSupportKHR(i, m_surface))
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

bool mvk::Device::CheckSwapChainAdequate(const vk::PhysicalDevice& device) const
{
	const auto formats = device.getSurfaceFormatsKHR(m_surface);
	const auto presentModes = device.getSurfacePresentModesKHR(m_surface);
	return !formats.empty() && !presentModes.empty();
}

bool mvk::Device::IsDeviceSuitable(const vk::PhysicalDevice& device) const
{
	return FindQueueFamilies(device).IsComplete()
		&& CheckDeviceExtensionSupport(device)
		&& CheckSwapChainAdequate(device);
}