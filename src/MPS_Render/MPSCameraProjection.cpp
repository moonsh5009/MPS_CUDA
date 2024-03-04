#include "stdafx.h"
#include "MPSCameraProjection.h"

#include "MPSCameraDef.h"
#include "GLMUtil.h"

using namespace mps::rndr;

CameraProjection::CameraProjection() :
	m_isDirty{ false },
	m_isOrtho{ false },
	m_height{ mps::rndr::DEFAULT_CAMERA_HEIGHT },
	m_aspectRatio{ mps::rndr::DEFAULT_CAMERA_ASPECTRATIO },
	m_zNear{ mps::rndr::DEFAULT_CAMERA_ZNEAR },
	m_zFar{ mps::rndr::DEFAULT_CAMERA_ZFAR }
{
}

constexpr void CameraProjection::SetHeight(float height) noexcept
{
	m_height = height;
	m_isDirty = false;
}

constexpr void CameraProjection::SetAspectRatio(float aspectRatio) noexcept
{
	m_aspectRatio = aspectRatio;
	m_isDirty = false;
}

constexpr void CameraProjection::SetAspectRatio(int width, int height) noexcept
{
	m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);
	m_isDirty = false;
}

constexpr void CameraProjection::SetZNear(float zNear) noexcept
{
	m_zNear = zNear;
	m_isDirty = false;
}

constexpr void CameraProjection::SetZFar(float zFar) noexcept
{
	m_zFar = zFar;
	m_isDirty = false;
}

bool CameraProjection::UpdateMatrix()
{
	if (m_isDirty)
		return false;

	if (m_isOrtho)
	{
		m_projMat = mps::util::Orthographic(m_height, m_aspectRatio, m_zNear, m_zFar);
	}
	else
	{
		m_projMat = mps::util::Perspective(m_height, m_aspectRatio, m_zNear, m_zFar);
	}
	m_projInvMat = glm::inverse(m_projMat);

	m_isDirty = true;
	return true;
}
