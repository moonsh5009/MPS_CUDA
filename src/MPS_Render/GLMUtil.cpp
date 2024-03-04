#include "stdafx.h"
#include "GLMUtil.h"

namespace mps
{
	namespace util
	{
		glm::fmat4 Translate(const glm::fvec3& v)
		{
			return
			{
				1.f,	0.f,	0.f,	0.f,
				0.f,	1.f,	0.f,	0.f,
				0.f,	0.f,	1.f,	0.f,
				v[0],	v[1],	v[2],	1.f
			};
		}
		glm::fmat4 Rotate(const glm::fvec3& zDir, const glm::fvec3& zUp, const glm::fvec3& zRight)
		{
			return
			{
				zRight[0],	zRight[1],	-zRight[2],	0.f,
				zUp[0],		zUp[1],		-zUp[2],	0.f,
				zDir[0],	zDir[1],	-zDir[2],	0.f,
				0.f,		0.f,		0.f,		1.f,
			};
		}

		glm::fmat4 Orthographic(
			const float left, const float right, const float bottom, const float top, const float zNear, const float zFar)
		{
			return
			{
				2.f / (right - left), 0.f, 0.f, 0.f,
				0.f, 2.f / (top - bottom), 0.f, 0.f,
				0.f, 0.f, 2.f / (zFar - zNear), 0.f,
				-(right + left) / (right - left), -(top + bottom) / (top - bottom), (zFar + zNear) / (zFar - zNear), 1.f
			};
		}

		glm::fmat4 Perspective(
			const float left, const float right, const float bottom, const float top, const float zNear, const float zFar)
		{
			return
			{
				2.f / (right - left), 0.f, 0.f, 0.f,
				0.f, 2.f / (top - bottom), 0.f, 0.f,
				(right + left) / (right - left), (top + bottom) / (top - bottom), (zFar + zNear) / (zFar - zNear), -1.f,
				0.f, 0.f, -2.f * zFar * zNear / (zFar - zNear),	0.f
			};
		}

		glm::fmat4 Orthographic(
			const float height, const float aspect, const float zNear, const float zFar)
		{
			return
			{
				1.f / (height * aspect), 0.f, 0.f, 0.f,
				0.f, 1.f / height, 0.f, 0.f,
				0.f, 0.f, 2.f / (zFar - zNear), 0.f,
				0.f, 0.f, (zFar + zNear) / (zFar - zNear), 1.f
			};
		}

		glm::fmat4 Perspective(
			const float height, const float aspect, const float zNear, const float zFar)
		{
			return
			{
				1.f / (aspect * height), 0.f, 0.f, 0.f,
				0.f, 1.f / height, 0.f, 0.f,
				0.f, 0.f, -(zFar + zNear) / (zFar - zNear), -1.f,
				0.f, 0.f, -2.f * zFar * zNear / (zFar - zNear), 0.f
			};
		}
	}
}