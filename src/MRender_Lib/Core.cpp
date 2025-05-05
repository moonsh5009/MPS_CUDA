#include "stdafx.h"
#include "Core.h"

mvk::Core::Core(HWND window)
{
	Initialize(window);
}

mvk::Core::~Core()
{
	Destroy();
}

void mvk::Core::Initialize(HWND window)
{
	try
	{
		m_pDevice = std::make_shared<Device>(window, GenerateInstance());
		m_pRenderContext = std::make_shared<RenderContext>(m_pDevice);
	}
	catch (std::runtime_error e)
	{
		const auto error = e.what();
	}
}

std::tuple<vk::Instance, vk::DebugUtilsMessengerEXT> mvk::Core::GenerateInstance()
{
	if (ENABLE_VALIDATION_LAYERS && !CheckValidationLayerSupport())
	{
		throw std::runtime_error("validation layers requested, but not available!");
	}

	std::string appName = "";
	std::string engineName = "";
	vk::ApplicationInfo appInfo(appName.c_str(), 1, engineName.c_str(), 1, VK_MAKE_VERSION(1, 0, 39));

	std::vector<const char*> layerNames;
	vk::InstanceCreateInfo createInfo({},
		&appInfo);

	const auto extensions = GetRequiredExtensions();
	createInfo.setPEnabledExtensionNames(extensions);

	vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;
	if (ENABLE_VALIDATION_LAYERS)
	{
		createInfo.setPEnabledLayerNames(VALIDATION_LAYERS);
		debugCreateInfo.setMessageSeverity(
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
		debugCreateInfo.setMessageType(
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
			vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
			vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance);
		debugCreateInfo.setPfnUserCallback(DebugCallback);
		createInfo.setPNext(&debugCreateInfo);
	}

	const auto instance = vk::createInstance(createInfo);
	if (!instance)
	{
		throw std::runtime_error("failed to create instance!");
	}

	vk::DebugUtilsMessengerEXT debugMessenger;
	if (ENABLE_VALIDATION_LAYERS)
	{
		/*debugMessenger = m_instance.createDebugUtilsMessengerEXT(debugCreateInfo);
		if (!debugMessenger)
		{
			throw std::runtime_error("failed to set up debug messenger!");
		}*/
	}

	return { instance, debugMessenger };
}

void mvk::Core::Destroy()
{
	m_pRenderContext.reset();
	m_pDevice.reset();
}

bool mvk::Core::CheckValidationLayerSupport() const
{
	const auto availableLayers = vk::enumerateInstanceLayerProperties();
	return std::all_of(VALIDATION_LAYERS.begin(), VALIDATION_LAYERS.end(), [&](const auto& layerName)
	{
		return std::any_of(availableLayers.begin(), availableLayers.end(), [&](const auto& layerProperties)
		{
			return std::strcmp(layerName, layerProperties.layerName) == 0;
		});
	});
}

std::vector<const char*> mvk::Core::GetRequiredExtensions() const
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	if (glfwExtensionCount == 0)
	{
		return { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
	}

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	if (ENABLE_VALIDATION_LAYERS)
	{
		extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}