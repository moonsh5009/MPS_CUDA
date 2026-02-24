#include "stdafx.h"
#include "CameraTransform.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace mcore::render
{
    CameraTransform::CameraTransform(const CameraTransform& src) noexcept
    {
        *this = src;
    }

    CameraTransform& CameraTransform::operator=(const CameraTransform& src) noexcept
    {
        m_position = src.m_position;
        m_rotation = src.m_rotation;
        m_scale = src.m_scale;

        m_posDirty = true;
        m_scaleDirty = true;
        m_rotationDirty = true;
        return *this;
    }

    bool CameraTransform::UpdateMatrix(bool emitSignal) noexcept
    {
        bool needToUpdateModelMat = false;

        needToUpdateModelMat |= UpdateTranslationMatrix();
        needToUpdateModelMat |= UpdateRotationMatrix();
        needToUpdateModelMat |= UpdateScaleMatrix();

        if (!needToUpdateModelMat)
            return false;

        OnUpdateModelMatrix();

        if (emitSignal)
            onUpdateMatrix->Dispatch();

        return true;
    }

    bool CameraTransform::UpdateTranslationMatrix() noexcept
    {
        if (!m_posDirty)
            return false;

        m_translationMat = glm::translate(m_position);
        m_posDirty = false;
        return true;
    }

    bool CameraTransform::UpdateScaleMatrix() noexcept
    {
        if (!m_scaleDirty)
            return false;

        m_scaleMat = glm::scale(m_scale);
        m_scaleDirty = false;
        return true;
    }

    bool CameraTransform::UpdateRotationMatrix() noexcept
    {
        if (!m_rotationDirty)
            return false;

        m_rotationMat = m_rotation.GetMatrix();
        m_rotationDirty = false;
        return true;
    }

    void CameraTransform::OnUpdateModelMatrix() noexcept
    {
        m_viewInvMat = m_translationMat * m_rotationMat * m_scaleMat;
        m_viewMat = glm::inverse(m_viewInvMat);
    }

    void CameraTransform::Orient(const glm::vec3& forward, const glm::vec3& referenceUp) noexcept
    {
        m_rotation.Orient(forward, referenceUp);
        m_rotationDirty = true;
    }

    void CameraTransform::LookAt(const glm::vec3& position, const glm::vec3& target, const glm::vec3& referenceUp) noexcept
    {
        SetPosition(position);
        Orient(target - position, referenceUp);
    }

    void CameraTransform::Orbit(float angle, const glm::vec3& pivot, const glm::vec3& axis, bool angleRotation) noexcept
    {
        const auto rotationQuat = glm::angleAxis(angle, glm::normalize(axis));
        m_position = ((rotationQuat * (m_position - pivot)) + pivot);
        m_posDirty = true;

        if (angleRotation)
        {
            m_rotation.RotateGlobal(angle, axis);
            m_rotationDirty = true;
        }
    }
}