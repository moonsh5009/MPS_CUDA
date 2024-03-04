#pragma once

#include <glm/glm.hpp>

namespace mps
{
	struct CameraParam
	{
		glm::fmat4 viewMat;
		glm::fmat4 projMat;
		glm::fmat4 viewInvMat;
		glm::fmat4 projInvMat;
	};
	struct LightParam
	{
		glm::fvec3 pos;
		glm::fvec4 color;
	};
}