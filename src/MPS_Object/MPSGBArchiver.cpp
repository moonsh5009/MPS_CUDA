#include "stdafx.h"
#include "MPSGBArchiver.h"

void mps::GBArchiver::Initalize()
{
	m_cameraBuffer.Resize(1);
	m_lightBuffer.Resize(1);
}

void mps::GBArchiver::UpdateCamera(const mps::CameraParam& camera)
{
	m_cameraBuffer.CopyFromHost(camera);
}

void mps::GBArchiver::UpdateLight(const mps::LightParam& light)
{
	m_lightBuffer.CopyFromHost(light);
}

void mps::GBArchiver::UpdateLightPos(const glm::vec3& lightPos)
{
	m_lightBuffer.CopyFromHost(lightPos);
}

void mps::GBArchiver::UpdateGravity(const REAL3& gravity)
{
	m_physicsBuffer.CopyFromHost(gravity);
}
