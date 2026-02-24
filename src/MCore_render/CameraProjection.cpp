#include "stdafx.h"
#include "CameraProjection.h"

#include "ProjectUtil.h"

#include <glm/gtx/transform.hpp>

using namespace std;
using namespace glm;

namespace mcore::render
{
    CameraProjection::CameraProjection() noexcept
    {
        SetProjectionType(constant::camera::DEFAULT_PROJ_TYPE);
        //SetProjectionType(ProjectionType::PERSPECTIVE);
    }

    CameraProjection::CameraProjection(const CameraProjection& src) noexcept
    {
        *this = src;
    }

    void CameraProjection::OnUpdateProjectionMatrix_Ortho() noexcept
    {
        const float yTop = m_height;
        const float yBottom = -yTop;

        const float xRight = (yTop * m_aspectRatio);
        const float xLeft = -xRight;

        m_projMat = ProjectUtil::Ortho(xLeft, xRight, yBottom, yTop, m_zNear, m_zFar);
    }

    void CameraProjection::OnUpdateProjectionMatrix_Perspective() noexcept
    {
        const float yTop = m_height;
        const float yBottom = -yTop;

        const float xRight = (yTop * m_aspectRatio);
        const float xLeft = -xRight;

        m_projMat = ProjectUtil::Frustum(xLeft, xRight, yBottom, yTop, m_zNear, m_zFar);
    }

    CameraProjection& CameraProjection::operator=(const CameraProjection& src) noexcept
    {
        m_projType = src.m_projType;
        m_onUpdateProjectionMatrix = src.m_onUpdateProjectionMatrix;

        m_height = src.m_height;
        m_aspectRatio = src.m_aspectRatio;
        m_zNear = src.m_zNear;
        m_zFar = src.m_zFar;
        m_dirty = true;

        return *this;
    }

    float CameraProjection::GetFovy() const noexcept
    {
        return (atanf(m_height / m_zNear) * 2.f);
    }

    bool CameraProjection::UpdateMatrix(bool emitSignal) noexcept
    {
        if (!m_dirty)
            return false;

        m_onUpdateProjectionMatrix(*this);
        m_projMat = m_captureMat * m_projMat;

        m_projInvMat = inverse(m_projMat);

        m_dirty = false;

        if (emitSignal)
            onUpdateMatrix->Dispatch();

        return true;
    }

    void CameraProjection::SetCaptureMatrix(const glm::vec2& translate, const glm::vec2& scale) noexcept
    {
        const auto translationMat = glm::translate(glm::vec3{ translate, 0.0f });
        const auto scaleMat = glm::scale(glm::vec3{ scale, 1.0f });
        m_captureMat = scaleMat * translationMat;
        m_dirty = true;
    }
}