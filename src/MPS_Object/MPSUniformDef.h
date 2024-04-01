#pragma once

#include "../MCUDA_Lib/MCUDAHelper.h"

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
		glm::vec3 pos;
		glm::fvec4 color;
	};
}