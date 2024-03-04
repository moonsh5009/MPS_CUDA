#pragma once

#include <glm/glm.hpp>

#include "HeaderPre.h"

namespace mps
{
	namespace util
	{
		glm::fmat4 __MY_EXT_API__ Translate(const glm::fvec3& v);
		glm::fmat4 __MY_EXT_API__ Rotate(const glm::fvec3& zDir, const glm::fvec3& zUp, const glm::fvec3& zRight);

		glm::fmat4 __MY_EXT_API__ Orthographic(const float left, const float right, const float bottom, const float top, const float zNear, const float zFar);
		glm::fmat4 __MY_EXT_API__ Orthographic(const float height, const float aspect, const float zNear, const float zFar);

		glm::fmat4 __MY_EXT_API__ Perspective(const float left, const float right, const float bottom, const float top, const float zNear, const float zFar);
		glm::fmat4 __MY_EXT_API__ Perspective(const float height, const float aspect, const float zNear, const float zFar);
	}
}

#include "HeaderPost.h"