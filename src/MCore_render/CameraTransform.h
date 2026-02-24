#pragma once

#include "../MCore_util/Signal.h"
#include "../MCore_util/Quaternion.h"
#include "CameraDef.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ CameraTransform
    {
    public:
        CameraTransform() = default;
        ~CameraTransform() = default;
        CameraTransform(const CameraTransform& src) noexcept;
        CameraTransform(CameraTransform&&) noexcept = default;
        CameraTransform& operator=(const CameraTransform& src) noexcept;
        CameraTransform& operator=(CameraTransform&&) noexcept = default;

        mcore::Signal<void()> onUpdateMatrix = mcore::MakeSignal<void()>();

        bool UpdateMatrix(bool emitSignal = true) noexcept;

        void Orient(const glm::vec3& forward, const glm::vec3& referenceUp = { 0., 0., 1. }) noexcept;
        void LookAt(const glm::vec3& position, const glm::vec3& target, const glm::vec3& referenceUp = { 0., 0., 1. }) noexcept;
        void Orbit(float angle, const glm::vec3& pivot, const glm::vec3& axis, bool angleRotation = true) noexcept;
        
        constexpr void MoveForward(float delta) noexcept;
        constexpr void MoveHorizontal(float delta) noexcept;
        constexpr void MoveVertical(float delta) noexcept;
        
        constexpr void SetPosition(const glm::vec3& position) noexcept;
        constexpr void SetScale(float uniformScale) noexcept;
        constexpr void SetScale(const glm::vec3& scale) noexcept;

        constexpr void SetRotation(const glm::quat& quaternion, bool normalization = true) noexcept;
        constexpr void SetRotation(const glm::vec3& eulerAngles) noexcept;
        void SetRotation(float angle, const glm::vec3& axis) noexcept;
        constexpr void SetRotation(const glm::mat3& rotationMatrix, bool normalization = true) noexcept;
        constexpr void SetRotation(const glm::mat4& rotationMatrix, bool normalization = true) noexcept;

        constexpr void RotateGlobal(const glm::vec3& eulerAngles) noexcept;
        void RotateGlobal(float angle, const glm::vec3& axis) noexcept;
        constexpr void RotateLocal(const glm::vec3& eulerAngles) noexcept;
        constexpr void RotateFPS(float pitch, float yaw, const glm::vec3& referenceUp = { 0., 0., 1. }) noexcept;

        constexpr void AdjustPosition(const glm::vec3& delta) noexcept;
        constexpr void AdjustScale(float uniformDelta) noexcept;
        constexpr void AdjustScale(const glm::vec3& delta) noexcept;

        constexpr const glm::vec3& GetPosition() const noexcept { return m_position; }
        constexpr const glm::vec3& GetScale() const noexcept { return m_scale; }
        constexpr const Quaternionf& GetRotation() const noexcept { return m_rotation; }

        constexpr const glm::vec4& GetForward() const noexcept { return m_rotationMat[2]; }
        constexpr const glm::vec4& GetHorizontal() const noexcept { return m_rotationMat[0]; }
        constexpr const glm::vec4& GetVertical() const noexcept { return m_rotationMat[1]; }
        constexpr const glm::mat4& GetViewMatrix() const noexcept { return m_viewMat; }
        constexpr const glm::mat4& GetViewInvMatrix() const noexcept { return m_viewInvMat; }

    private:
        glm::vec3 m_position{ 0., 0., 0. };
        glm::vec3 m_scale{ 1., 1., 1. };
        Quaternion<float> m_rotation;

        glm::mat4 m_translationMat{ 1. };
        glm::mat4 m_scaleMat{ 1. };
        glm::mat4 m_rotationMat{ 1. };

        glm::mat4 m_viewMat{ 1. };
        glm::mat4 m_viewInvMat{ 1. };

        bool m_posDirty = true;
        bool m_scaleDirty = true;
        bool m_rotationDirty = true;

        constexpr bool IsPosDirty() const noexcept { return m_posDirty; }
        constexpr bool IsScaleDirty() const noexcept { return m_scaleDirty; }
        constexpr bool IsRotationDirty() const noexcept { return m_rotationDirty; }

        bool UpdateTranslationMatrix() noexcept;
        bool UpdateScaleMatrix() noexcept;
        bool UpdateRotationMatrix() noexcept;
        void OnUpdateModelMatrix() noexcept;
    };

    constexpr void CameraTransform::MoveForward(float delta) noexcept
    {
        return AdjustPosition(GetForward() * delta);
    }

    constexpr void CameraTransform::MoveHorizontal(float delta) noexcept
    {
        return AdjustPosition(GetHorizontal() * delta);
    }

    constexpr void CameraTransform::MoveVertical(float delta) noexcept
    {
        return AdjustPosition(GetVertical() * delta);
    }

    constexpr void CameraTransform::SetPosition(const glm::vec3& position) noexcept
    {
        m_position = position;
        m_posDirty = true;
    }

    constexpr void CameraTransform::SetScale(float uniformScale) noexcept
    {
        SetScale(glm::vec3{ uniformScale });
    }

    constexpr void CameraTransform::SetScale(const glm::vec3& scale) noexcept
    {
        m_scale = glm::clamp(scale, constant::camera::MIN_SCALE, constant::camera::MAX_SCALE);
        m_scaleDirty = true;
    }

    constexpr void CameraTransform::SetRotation(const glm::quat& quaternion, bool normalization) noexcept
    {
        m_rotation.Set(quaternion);
        if (normalization)
            m_rotation.Normalize();
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::SetRotation(const glm::vec3& eulerAngles) noexcept
    {
        m_rotation.Set(eulerAngles);
        m_rotationDirty = true;
    }

    inline void CameraTransform::SetRotation(float angle, const glm::vec3& axis) noexcept
    {
        m_rotation.Set(angle, axis);
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::SetRotation(const glm::mat3& rotationMatrix, bool normalization) noexcept
    {
        m_rotation.Set(rotationMatrix);
        if (normalization)
            m_rotation.Normalize();
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::SetRotation(const glm::mat4& rotationMatrix, bool normalization) noexcept
    {
        m_rotation.Set(rotationMatrix);
        if (normalization)
            m_rotation.Normalize();
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::RotateGlobal(const glm::vec3& eulerAngles) noexcept
    {
        m_rotation.RotateGlobal(eulerAngles);
        m_rotationDirty = true;
    }

    inline void CameraTransform::RotateGlobal(float angle, const glm::vec3& axis) noexcept
    {
        m_rotation.RotateGlobal(angle, axis);
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::RotateLocal(const glm::vec3& eulerAngles) noexcept
    {
        m_rotation.RotateLocal(eulerAngles);
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::RotateFPS(float pitch, float yaw, const glm::vec3& referenceUp) noexcept
    {
        m_rotation.RotateFPS(pitch, yaw, referenceUp);
        m_rotationDirty = true;
    }

    constexpr void CameraTransform::AdjustPosition(const glm::vec3& delta) noexcept
    {
        SetPosition(m_position + delta);
    }

    constexpr void CameraTransform::AdjustScale(float uniformDelta) noexcept
    {
        AdjustScale(glm::vec3{ uniformDelta });
    }

    constexpr void CameraTransform::AdjustScale(const glm::vec3& delta) noexcept
    {
        SetScale(m_scale + delta);
    }
}

#include "HeaderPost.h"