#pragma once

#define VULKAN_HPP_NO_SPACESHIP_OPERATOR
#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan/vulkan.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <optional>
#include <array>
#include <set>
#include <string>
#include <string_view>
#include <span>

#include "Logger.h"

namespace mvk
{
#ifdef _DEBUG
	constexpr bool ENABLE_VALIDATION_LAYERS = true;
#else
	constexpr bool ENABLE_VALIDATION_LAYERS = false;
#endif
	constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

	constexpr std::string_view SHADER_DIRECTORY = "./shader/";
	constexpr std::string_view PIPELINE_CACHE_DIRECTORY = "./cache/render_pipeline/";

	constexpr std::string GetResourcePath(const std::string_view& path)
	{
		return "~/" + std::string(path);
	}

    enum class QueueType
    {
        GRAPHIC = 0,
        COMPUTE,
        TRANSFER,
        Size,
    };

    enum class CommandType
    {
        PREBAKED,
        REUSABLE,
        DYNAMIC,
        Size
    };

    struct DeviceFeatures
    {
        bool dynamicRendering = false;
        bool synchronization2 = false;
        bool timelineSemaphore = false;
        bool maintenance4 = false;
        bool bufferDeviceAddress = false;
        bool descriptorIndexing = false;
        bool multiview = false;
        vk::SampleCountFlagBits maxMsaaSamples = vk::SampleCountFlagBits::e1;
    };

    struct GPUInfo
    {
        vk::PhysicalDevice device;
        vk::PhysicalDeviceProperties properties;
        vk::PhysicalDeviceFeatures features;
        vk::PhysicalDeviceMemoryProperties memoryProperties;
        DeviceFeatures supportedFeatures;
        uint32_t score = 0;
        std::string scoreBreakdown;

        bool operator>(const GPUInfo& other) const
        {
            return score > other.score;
        }
    };

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> present;
        std::optional<uint32_t> compute;
        std::optional<uint32_t> transfer;

        bool IsComplete() const
        {
            return graphics.has_value() && present.has_value();
        }

        std::vector<uint32_t> GetAllIndices() const
        {
            if (!IsComplete()) return {};

            std::set<uint32_t> uniqueIndices;

            uniqueIndices.insert(graphics.value());
            uniqueIndices.insert(present.value());

            if (compute.has_value()) uniqueIndices.insert(compute.value());
            if (transfer.has_value()) uniqueIndices.insert(transfer.value());

            return std::vector<uint32_t>(uniqueIndices.begin(), uniqueIndices.end());
        }

        std::vector<uint32_t> GetSwapChainQueueIndices() const
        {
            if (!IsComplete()) return {};

            std::set<uint32_t> uniqueIndices;
            uniqueIndices.insert(graphics.value());
            uniqueIndices.insert(present.value());
            return std::vector<uint32_t>(uniqueIndices.begin(), uniqueIndices.end());
        }

        constexpr void reset()
        {
            graphics.reset();
            present.reset();
            compute.reset();
            transfer.reset();
        }

        constexpr uint32_t Get(QueueType type) const
        {
            switch (type)
            {
            case QueueType::GRAPHIC:
                return GetGraphics();
            case QueueType::COMPUTE:
                return GetCompute();
            case QueueType::TRANSFER:
                return GetTransfer();
            default:
                return GetGraphics();
            }
        }
        constexpr uint32_t GetGraphics() const { return graphics.value_or(0); }
        constexpr uint32_t GetPresent() const { return present.value_or(0); }
        constexpr uint32_t GetCompute() const { return compute.value_or(graphics.value_or(0)); }
        constexpr uint32_t GetTransfer() const { return transfer.value_or(graphics.value_or(0)); }
    };

    constexpr uint32_t GetFormatSize(vk::Format format)
    {
        switch (format)
        {
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Snorm:
        case vk::Format::eR8Uint:
        case vk::Format::eR8Sint:
        case vk::Format::eR8Srgb:
            return 1;

        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR8G8Uint:
        case vk::Format::eR8G8Sint:
        case vk::Format::eR8G8Srgb:
        case vk::Format::eR16Unorm:
        case vk::Format::eR16Snorm:
        case vk::Format::eR16Uint:
        case vk::Format::eR16Sint:
        case vk::Format::eR16Sfloat:
            return 2;

        case vk::Format::eR8G8B8Unorm:
        case vk::Format::eR8G8B8Snorm:
        case vk::Format::eR8G8B8Uint:
        case vk::Format::eR8G8B8Sint:
        case vk::Format::eR8G8B8Srgb:
        case vk::Format::eB8G8R8Unorm:
        case vk::Format::eB8G8R8Snorm:
        case vk::Format::eB8G8R8Uint:
        case vk::Format::eB8G8R8Sint:
        case vk::Format::eB8G8R8Srgb:
            return 3;

        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Snorm:
        case vk::Format::eR8G8B8A8Uint:
        case vk::Format::eR8G8B8A8Sint:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Snorm:
        case vk::Format::eB8G8R8A8Uint:
        case vk::Format::eB8G8R8A8Sint:
        case vk::Format::eB8G8R8A8Srgb:
        case vk::Format::eA8B8G8R8UnormPack32:
        case vk::Format::eA8B8G8R8SnormPack32:
        case vk::Format::eA8B8G8R8UintPack32:
        case vk::Format::eA8B8G8R8SintPack32:
        case vk::Format::eA8B8G8R8SrgbPack32:
        case vk::Format::eR16G16Unorm:
        case vk::Format::eR16G16Snorm:
        case vk::Format::eR16G16Uint:
        case vk::Format::eR16G16Sint:
        case vk::Format::eR16G16Sfloat:
        case vk::Format::eR32Uint:
        case vk::Format::eR32Sint:
        case vk::Format::eR32Sfloat:
        case vk::Format::eD32Sfloat:
        case vk::Format::eD24UnormS8Uint:
            return 4;

        case vk::Format::eA2R10G10B10UnormPack32:
        case vk::Format::eA2R10G10B10SnormPack32:
        case vk::Format::eA2R10G10B10UintPack32:
        case vk::Format::eA2R10G10B10SintPack32:
        case vk::Format::eA2B10G10R10UnormPack32:
        case vk::Format::eA2B10G10R10SnormPack32:
        case vk::Format::eA2B10G10R10UintPack32:
        case vk::Format::eA2B10G10R10SintPack32:
        case vk::Format::eR5G6B5UnormPack16:
        case vk::Format::eB5G6R5UnormPack16:
        case vk::Format::eR5G5B5A1UnormPack16:
        case vk::Format::eB5G5R5A1UnormPack16:
        case vk::Format::eA1R5G5B5UnormPack16:
        case vk::Format::eR4G4B4A4UnormPack16:
        case vk::Format::eB4G4R4A4UnormPack16:
            return 4;

        case vk::Format::eR16G16B16A16Unorm:
        case vk::Format::eR16G16B16A16Snorm:
        case vk::Format::eR16G16B16A16Uint:
        case vk::Format::eR16G16B16A16Sint:
        case vk::Format::eR16G16B16A16Sfloat:
        case vk::Format::eR32G32Uint:
        case vk::Format::eR32G32Sint:
        case vk::Format::eR32G32Sfloat:
        case vk::Format::eD32SfloatS8Uint:
            return 8;

        case vk::Format::eR32G32B32Uint:
        case vk::Format::eR32G32B32Sint:
        case vk::Format::eR32G32B32Sfloat:
            return 12;

        case vk::Format::eR32G32B32A32Uint:
        case vk::Format::eR32G32B32A32Sint:
        case vk::Format::eR32G32B32A32Sfloat:
            return 16;

        case vk::Format::eBc1RgbUnormBlock:
        case vk::Format::eBc1RgbSrgbBlock:
        case vk::Format::eBc1RgbaUnormBlock:
        case vk::Format::eBc1RgbaSrgbBlock:
        case vk::Format::eBc4UnormBlock:
        case vk::Format::eBc4SnormBlock:
            return 8;

        case vk::Format::eBc2UnormBlock:
        case vk::Format::eBc2SrgbBlock:
        case vk::Format::eBc3UnormBlock:
        case vk::Format::eBc3SrgbBlock:
        case vk::Format::eBc5UnormBlock:
        case vk::Format::eBc5SnormBlock:
        case vk::Format::eBc6HUfloatBlock:
        case vk::Format::eBc6HSfloatBlock:
        case vk::Format::eBc7UnormBlock:
        case vk::Format::eBc7SrgbBlock:
            return 16;

        case vk::Format::eAstc4x4UnormBlock:
        case vk::Format::eAstc4x4SrgbBlock:
        case vk::Format::eAstc5x4UnormBlock:
        case vk::Format::eAstc5x4SrgbBlock:
        case vk::Format::eAstc5x5UnormBlock:
        case vk::Format::eAstc5x5SrgbBlock:
        case vk::Format::eAstc6x5UnormBlock:
        case vk::Format::eAstc6x5SrgbBlock:
        case vk::Format::eAstc6x6UnormBlock:
        case vk::Format::eAstc6x6SrgbBlock:
        case vk::Format::eAstc8x5UnormBlock:
        case vk::Format::eAstc8x5SrgbBlock:
        case vk::Format::eAstc8x6UnormBlock:
        case vk::Format::eAstc8x6SrgbBlock:
        case vk::Format::eAstc8x8UnormBlock:
        case vk::Format::eAstc8x8SrgbBlock:
        case vk::Format::eAstc10x5UnormBlock:
        case vk::Format::eAstc10x5SrgbBlock:
        case vk::Format::eAstc10x6UnormBlock:
        case vk::Format::eAstc10x6SrgbBlock:
        case vk::Format::eAstc10x8UnormBlock:
        case vk::Format::eAstc10x8SrgbBlock:
        case vk::Format::eAstc10x10UnormBlock:
        case vk::Format::eAstc10x10SrgbBlock:
        case vk::Format::eAstc12x10UnormBlock:
        case vk::Format::eAstc12x10SrgbBlock:
        case vk::Format::eAstc12x12UnormBlock:
        case vk::Format::eAstc12x12SrgbBlock:
            return 16;

        case vk::Format::eD16Unorm:
            return 2;
        case vk::Format::eD16UnormS8Uint:
            return 3;
        case vk::Format::eS8Uint:
            return 1;

        default:
            return 0;
        }
    }
}