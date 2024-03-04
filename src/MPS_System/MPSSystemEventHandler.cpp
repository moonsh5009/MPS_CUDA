#include "stdafx.h"
#include "MPSSystemEventHandler.h"

mps::SystemEventHandler::SystemEventHandler()
{
}

void mps::SystemEventHandler::Initalize()
{
}

void mps::SystemEventHandler::LMouseDown(glm::ivec2 curPos, int mods)
{
    OnLMouseDown(curPos, mods);
}

void mps::SystemEventHandler::RMouseDown(glm::ivec2 curPos, int mods)
{
    OnRMouseDown(curPos, mods);
}

void mps::SystemEventHandler::WMouseDown(glm::ivec2 curPos, int mods)
{
    OnWMouseDown(curPos, mods);
}

void mps::SystemEventHandler::LMouseUp(glm::ivec2 curPos, int mods)
{
    OnLMouseUp(curPos, mods);
}

void mps::SystemEventHandler::RMouseUp(glm::ivec2 curPos, int mods)
{
    OnRMouseUp(curPos, mods);
}

void mps::SystemEventHandler::WMouseUp(glm::ivec2 curPos, int mods)
{
    OnWMouseUp(curPos, mods);
}

void mps::SystemEventHandler::MouseMove(glm::ivec2 curPos)
{
    OnMouseMove(curPos);
}

void mps::SystemEventHandler::MouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods)
{
    OnMouseWheel(curPos, offset, mods);
}

void mps::SystemEventHandler::KeyDown(int key, int mods)
{
    OnKeyDown(key, mods);
}

void mps::SystemEventHandler::KeyUp(int key, int mods)
{
    OnKeyUp(key, mods);
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

void mps::SystemEventHandler::OnLMouseDown(glm::ivec2 curPos, int mods)
{
}

void mps::SystemEventHandler::OnRMouseDown(glm::ivec2 curPos, int mods)
{
}

void mps::SystemEventHandler::OnWMouseDown(glm::ivec2 curPos, int mods)
{
}

void mps::SystemEventHandler::OnLMouseUp(glm::ivec2 curPos, int mods)
{
}

void mps::SystemEventHandler::OnRMouseUp(glm::ivec2 curPos, int mods)
{
}

void mps::SystemEventHandler::OnWMouseUp(glm::ivec2 curPos, int mods)
{
}

void mps::SystemEventHandler::OnMouseMove(glm::ivec2 curPos)
{
}

void mps::SystemEventHandler::OnMouseWheel(glm::ivec2 curPos, glm::ivec2 offset, int mods)
{
}

void mps::SystemEventHandler::OnKeyDown(int key, int mods)
{
}

void mps::SystemEventHandler::OnKeyUp(int key, int mods)
{
}

void mps::SystemEventHandler::OnResize(int width, int height)
{
}