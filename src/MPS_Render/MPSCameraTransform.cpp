#include "stdafx.h"
#include "MPSCameraTransform.h"

#include "GLMUtil.h"

using namespace mps::rndr;

CameraTransform::CameraTransform() :
	m_isDirty{ false },
	m_transMat{ 1.0f },
	m_rotateMat{ 1.0f }
{
	SetPosition({ 0.f, 0.f, -1.f });
	SetXDir({ 1.0f, 0.0f, 0.0f });
	SetYDir({ 0.0f, 1.0f, 0.0f });
	SetZDir({ 0.0f, 0.0f, 1.0f });

	m_isDirty = false;
}
void CameraTransform::Rotate(float x, float y, float radian)
{
#if 0
	if (x != 0.0f)
	{
		float cx = cosf(x * radian);
		float sx = sinf(x * radian);
		glm::fmat4 rotXMat =
		{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, cx, -sx, 0.0f,
			0.0f, sx, cx, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		m_rotateMat = m_rotateMat * rotXMat;
		m_isDirty = false;
	}
	if (y != 0.0f)
	{
		float cy = cosf(y * radian);
		float sy = sinf(y * radian);
		glm::fmat4 rotYMat =
		{
			cy, 0.0f, sy, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			-sy, 0.0f, cy, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		m_rotateMat = rotYMat * m_rotateMat;
		m_isDirty = false;
	}
#else
	if (x != 0.0f)
	{
		float cx = cosf(x * radian);
		float sx = sinf(x * radian);
		glm::fvec4 zDir = { GetZDir(), 1.0f };
		glm::fmat4 rotXMat =
		{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, cx, -sx, 0.0f,
			0.0f, sx, cx, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		m_transMat[3] = m_rotateMat * rotXMat * inverse(m_rotateMat) * m_transMat[3];
		m_rotateMat = m_rotateMat * rotXMat;
		m_isDirty = false;
	}
	if (y != 0.0f)
	{
		float cy = cosf(y * radian);
		float sy = sinf(y * radian);
		glm::fmat4 rotYMat =
		{
			cy, 0.0f, sy, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			-sy, 0.0f, cy, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		m_transMat[3] = rotYMat * m_transMat[3];
		m_rotateMat = rotYMat * m_rotateMat;
		m_isDirty = false;
	}
#endif
}

bool CameraTransform::UpdateMatrix()
{
	if (m_isDirty)
		return false;

	m_viewInvMat = m_transMat * m_rotateMat;
	m_viewMat = glm::inverse(m_viewInvMat);

	m_isDirty = true;
	return true;
}