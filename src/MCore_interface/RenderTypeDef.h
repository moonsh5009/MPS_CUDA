#pragma once

#include "../MCore_util/VulkanDef.h"

namespace mcore::render
{
	constexpr auto RENDER_ID_TEXTURE_FORMAT = vk::Format::eR32G32B32A32Uint;

	enum class RenderUniformType
	{
		RENDER_CONFIG = 0,
		CAMERA,
		LIGHT,

		Size
	};

	enum class RenderStorageType
	{
		Size
	};

	enum class RenderTargetType
	{
		MSAA = 0,
		ID,

		Size
	};

	enum class RenderModelType
	{
		None = 0,
		
		Mesh,
		AABB,
		FixedElement,

		Size
	};

	enum class RenderPassType
	{
		Prefix = 0,
		Opaque,
		Transparent,
		ID,
		Post,

		Size
	};


	enum class CameraDirectionType : int
	{
		FRONT = 0,
		RIGHT,
		BOTTOM,
		PERSPECTIVE,
		NUM
	};
}