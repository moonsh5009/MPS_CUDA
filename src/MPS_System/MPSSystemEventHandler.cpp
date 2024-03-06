#include "stdafx.h"
#include "MPSSystemEventHandler.h"

mps::SystemEventHandler::SystemEventHandler()
{
}

void mps::SystemEventHandler::Initalize()
{
}

void mps::SystemEventHandler::LMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
    OnLMouseDown(flag, curPos);
}

void mps::SystemEventHandler::RMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
    OnRMouseDown(flag, curPos);
}

void mps::SystemEventHandler::WMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
    OnWMouseDown(flag, curPos);
}

void mps::SystemEventHandler::LMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
    OnLMouseUp(flag, curPos);
}

void mps::SystemEventHandler::RMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
    OnRMouseUp(flag, curPos);
}

void mps::SystemEventHandler::WMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
    OnWMouseUp(flag, curPos);
}

void mps::SystemEventHandler::MouseMove(mevent::Flag flag, glm::ivec2 curPos)
{
    OnMouseMove(flag, curPos);
}

void mps::SystemEventHandler::MouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos)
{
    OnMouseWheel(flag, offset, curPos);
}

void mps::SystemEventHandler::KeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
    OnKeyDown(key, repCnt, flag);
}

void mps::SystemEventHandler::KeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
    OnKeyUp(key, repCnt, flag);
}

void mps::SystemEventHandler::Resize(int width, int height)
{
    OnResize(width, height);
}

void mps::SystemEventHandler::Update()
{
    OnUpdate();
}

void mps::SystemEventHandler::Draw()
{
    OnDraw();
}

void mps::SystemEventHandler::OnLMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnRMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnLMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnRMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnMouseMove(mevent::Flag flag, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnKeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
}

void mps::SystemEventHandler::OnKeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
}

void mps::SystemEventHandler::OnResize(int width, int height)
{
}