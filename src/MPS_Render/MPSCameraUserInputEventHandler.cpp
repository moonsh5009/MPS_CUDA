#include "stdafx.h"
#include "MPSCameraUserInputEventHandler.h"

using namespace mps::rndr;

CameraUserInputEventHandler::CameraUserInputEventHandler(Camera* pCamera) :
	m_pCamera{ pCamera },
	m_isDown{ false },
	m_isRotate{ false },
	m_startPos{ 0, 0 }
{
}

void CameraUserInputEventHandler::OnWMouseDown(glm::ivec2 curPos, int mods)
{
	if (m_isDown)
		return;

	m_isRotate = mods & mevent::MMods::Ctrl;
	m_startPos = curPos;
	m_isDown = true;
}

void CameraUserInputEventHandler::OnWMouseUp(glm::ivec2 curPos, int mods)
{
	if (!m_isDown)
		return;

	m_isDown = false;
}

void CameraUserInputEventHandler::OnMouseMove(glm::ivec2 curPos)
{
	if (!m_isDown)
		return;

	const glm::fvec2 dif = m_startPos - curPos;
	m_startPos = curPos;

	if (m_isRotate)
	{
		m_pCamera->GetTransform()->Rotate(dif.y, dif.x, 0.002f);
	}
	else
	{
		m_pCamera->Move(0.008f * dif);
	}
}

void CameraUserInputEventHandler::OnMouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods)
{
	m_pCamera->MoveForward(0.24f * offset.y);
}
