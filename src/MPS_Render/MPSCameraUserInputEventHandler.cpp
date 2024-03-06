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

void CameraUserInputEventHandler::OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
	if (m_isDown)
		return;

	m_isRotate = (flag & mevent::Flag::Ctrl) != mevent::Flag::None;
	m_startPos = curPos;
	m_isDown = true;
}

void CameraUserInputEventHandler::OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
	if (!m_isDown)
		return;

	m_isDown = false;
}

void CameraUserInputEventHandler::OnMouseMove(mevent::Flag flag, glm::ivec2 curPos)
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

void CameraUserInputEventHandler::OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos)
{
	m_pCamera->MoveForward(0.24f * offset.y);
}
