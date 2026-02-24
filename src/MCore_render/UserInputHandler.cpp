#include "stdafx.h"
#include "UserInputHandler.h"

#include "../MCore_interface/IScene.h"

using namespace mcore::render;

UserInputHandler::UserInputHandler(IScene* pScene) noexcept :
	IUserInputHandler{ pScene },
	m_pCameraUserInputHandler{ std::make_unique<CameraUserInputHandler>(pScene) }
{}

void UserInputHandler::OnMouseUp(MouseEventInfo info)
{
	m_pCameraUserInputHandler->OnMouseUp(info);
}

void UserInputHandler::OnMouseDown(MouseEventInfo info)
{
	m_pCameraUserInputHandler->OnMouseDown(info);
}

void UserInputHandler::OnMouseMove(MouseEventInfo info)
{
	m_pCameraUserInputHandler->OnMouseMove(info);
}

bool UserInputHandler::OnMouseWheel(MouseEventInfo info)
{
	return m_pCameraUserInputHandler->OnMouseWheel(info);
}

void UserInputHandler::OnKeyUp(KeyEventInfo info)
{
	m_pCameraUserInputHandler->OnKeyUp(info);
}

void UserInputHandler::OnKeyDown(KeyEventInfo info)
{
	m_pCameraUserInputHandler->OnKeyDown(info);
}

void UserInputHandler::OnResize(unsigned width, unsigned height)
{
	GetScene()->OnResize(width, height);
}

void UserInputHandler::OnDraw()
{
	GetScene()->Draw();
}
