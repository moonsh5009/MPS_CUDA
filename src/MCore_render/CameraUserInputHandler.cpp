#include "stdafx.h"
#include "CameraUserInputHandler.h"

#include "../MCore_interface/IScene.h"

#include "CameraUtil.h"
#include "CameraUniform.h"

#include "IDRenderTarget.h"

#include <chrono>

using namespace mcore::render;

namespace
{
    constexpr unsigned WHEEL_SCROLL = 0x1000;
}

CameraUserInputHandler::CameraUserInputHandler(IScene* pScene) noexcept :
    m_pScene{ pScene },
    m_bMoving{ false },
    m_prevMousePoint{ 0, 0 },
    m_posMovePivot{ 0.0, 0.0, 0.0 }
{
}

void CameraUserInputHandler::OnMouseUp(MouseEventInfo info)
{
    if (CheckFlags(info.button, MouseButtonBits::Middle))
    {
        m_bMoving = false;
        m_bRotating = false;
    }
}

void CameraUserInputHandler::OnMouseDown(MouseEventInfo info)
{
    if (CheckFlags(info.button, MouseButtonBits::Middle))
    {
        m_prevMousePoint = { info.x, info.y };

        UpdateMovePivot();
    }
}

void CameraUserInputHandler::OnMouseMove(MouseEventInfo info)
{
    if (!CheckFlags(info.button, MouseButtonBits::Middle))
        return;

    m_bMoving = true;
    m_bRotating = CheckFlags(info.modifiers, MouseModifierBits::Ctrl);

    if (m_bRotating)
        Rotate({ info.x, info.y });
    else
        Translate({ info.x, info.y });
    m_prevMousePoint = { info.x, info.y };
}

bool CameraUserInputHandler::OnMouseWheel(MouseEventInfo info)
{
    if (m_bMoving) return false;

    m_bScroll = true;

    const auto pCameraUniform = GetScene()->GetUniform<CameraUniform>();
    auto& camera = pCameraUniform->GetCamera();
    auto& projInfo = camera.GetProjection();

    auto delta = static_cast<float>(info.wheelDelta);
    if (CheckFlags(info.modifiers, MouseModifierBits::Shift))       delta *= 10.f;
    else if (CheckFlags(info.modifiers, MouseModifierBits::Ctrl))   delta *= .1f;
    if (projInfo.GetProjectionType() == ProjectionType::ORTHO)
    {
        Zoom(delta / 2400.f, { info.x, info.y });
    }
    else
    {
        Advance(delta / 120.f, { info.x, info.y });
    }

    m_bScroll = false;
    return true;
}

void CameraUserInputHandler::OnKeyUp(KeyEventInfo info)
{
    if (info.keyCode == VK_CONTROL)
    {
        m_bRotating = false;
    }
}

void CameraUserInputHandler::OnKeyDown(KeyEventInfo info)
{
    if (info.keyCode == VK_CONTROL)
    {
        m_bRotating = false;
    }
}

glm::ivec2 CameraUserInputHandler::GetMouseViewportPos() const noexcept
{
    return ConvertMousePosToScreenPos(m_prevMousePoint);
}

glm::ivec2 CameraUserInputHandler::ConvertMousePosToScreenPos(const std::pair<int, int>& mousePos) const noexcept
{
    const auto& viewport = GetScene()->GetViewport();
    return { mousePos.first, viewport.w - (mousePos.second + 1) };
}

void CameraUserInputHandler::UpdateMovePivot() noexcept
{
    const auto& viewport = GetScene()->GetViewport();
    const auto pCameraUniform = GetScene()->GetUniform<CameraUniform>();
    const auto& camera = pCameraUniform->GetCamera();

    const auto& targetScreenPos = ConvertMousePosToScreenPos(m_prevMousePoint);

    const auto pIDRenderTarget = GetScene()->GetRenderingEngine()->GetTarget<IDRenderTarget>();
	m_depthMovePivot = pIDRenderTarget->GetDepth(static_cast<uint32_t>(m_prevMousePoint.first), static_cast<uint32_t>(m_prevMousePoint.second));

    if (glm::epsilonEqual(m_depthMovePivot, 0.f, glm::epsilon<float>()))
    {
        m_depthMovePivot = std::clamp(CameraUtil::Project(camera, m_posMovePivot, viewport).z, 0.0f, 1.0f);
    }
    else
    {
        m_posMovePivot = CameraUtil::Unproject(camera, { targetScreenPos, m_depthMovePivot }, viewport);
    }
}

void CameraUserInputHandler::Translate(const std::pair<int, int>& mousePoint)
{
    const auto& viewport = GetScene()->GetViewport();
    const auto pCameraUniform = GetScene()->GetUniform<CameraUniform>();
    auto& camera = pCameraUniform->GetCamera();

    const auto& prevWorldPos = CameraUtil::Unproject(camera, { ConvertMousePosToScreenPos(m_prevMousePoint), m_depthMovePivot }, viewport);
    const auto& curWorldPos = CameraUtil::Unproject(camera, { ConvertMousePosToScreenPos(mousePoint), m_depthMovePivot }, viewport);

    camera.GetTransform().AdjustPosition(prevWorldPos - curWorldPos);
    camera.UpdateMatrix();
}

void CameraUserInputHandler::Rotate(const std::pair<int, int>& mousePoint)
{
    constexpr auto SENSITIVITY = .0018f;
    const auto& delta = SENSITIVITY * glm::vec2(m_prevMousePoint.first - mousePoint.first, m_prevMousePoint.second - mousePoint.second);

    const auto& viewport = GetScene()->GetViewport();
    const auto pCameraUniform = GetScene()->GetUniform<CameraUniform>();
    auto& camera = pCameraUniform->GetCamera();

    auto& transform = camera.GetTransform();
    transform.Orbit(delta.y, m_posMovePivot, transform.GetHorizontal());
    transform.Orbit(delta.x, m_posMovePivot, { 0., 0., 1. });
    camera.UpdateMatrix();
}

void CameraUserInputHandler::Zoom(float delta, const std::pair<int, int>& mousePoint)
{
    const auto& posScreen = ConvertMousePosToScreenPos(mousePoint);

    const auto& viewport = GetScene()->GetViewport();
    const auto pCameraUniform = GetScene()->GetUniform<CameraUniform>();
    auto& camera = pCameraUniform->GetCamera();

    const auto& prevZoomPivot = CameraUtil::Unproject(camera, { posScreen, 0.0 }, viewport);

    auto& projection = camera.GetProjection();
    projection.AdjustHeight(projection.GetHeight() * -delta);
    projection.UpdateMatrix(false);

    const auto& currZoomPivot = CameraUtil::Unproject(camera, { posScreen, 0.0 }, viewport);

    camera.GetTransform().AdjustPosition(prevZoomPivot - currZoomPivot);
    camera.UpdateMatrix();
}

void CameraUserInputHandler::Advance(float delta, const std::pair<int, int>& mousePoint)
{
    const auto& viewport = GetScene()->GetViewport();
    const auto pCameraUniform = GetScene()->GetUniform<CameraUniform>();
    auto& camera = pCameraUniform->GetCamera();

    const auto& posScreen = ConvertMousePosToScreenPos(mousePoint);
    auto& transform = camera.GetTransform();

    const auto& advancePivot = CameraUtil::Unproject(camera, { posScreen, 0.0 }, viewport);
    transform.AdjustPosition(glm::normalize(advancePivot - transform.GetPosition()) * delta);
    camera.UpdateMatrix();
}