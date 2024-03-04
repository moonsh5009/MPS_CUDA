#include "stdafx.h"
#include "MPSCamera.h"

#include "../MPS_Object/MPSGBArchiver.h"

using namespace mps::rndr;

Camera::Camera() :
	m_pTransform{ std::make_unique<CameraTransform>() },
	m_pProjection{ std::make_unique<CameraProjection>() }
{
}

void Camera::Update(mps::GBArchiver* pGBArchiver)
{
	if (!UpdateMatrix())
		return;

	m_param.viewMat = m_pTransform->GetMatrix();
	m_param.projMat = m_pProjection->GetMatrix();
	m_param.viewInvMat = m_pTransform->GetInvMatrix();
	m_param.projInvMat = m_pProjection->GetInvMatrix();
	pGBArchiver->UpdateCamera(m_param);

	m_updateListener.Notify();
}

bool Camera::UpdateMatrix()
{
	bool isNotDirty = false;
	isNotDirty |= m_pTransform->UpdateMatrix();
	isNotDirty |= m_pProjection->UpdateMatrix();
	return isNotDirty;
}

void Camera::MoveForward(const float x)
{
	auto pos = m_pTransform->GetPosition();
	pos += m_pTransform->GetZDir() * x;
	//pos += glm::fvec3{ 0.0, 0.0, x };
	m_pTransform->SetPosition(pos);
}

void Camera::Move(const glm::fvec2 dif)
{
	auto pos = m_pTransform->GetPosition();
	pos += m_pTransform->GetXDir() * dif.x;
	pos += m_pTransform->GetYDir() * dif.y;
	//pos += glm::fvec3{ dif.x, dif.y, 0.0 };
	m_pTransform->SetPosition(pos);
}