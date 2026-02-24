#pragma once

#include "CameraDef.h"
#include "../MCore_util/Signal.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ CameraProjection
    {
    public:
        mcore::Signal<void()> onUpdateMatrix = mcore::MakeSignal<void()>();

        CameraProjection() noexcept;
        ~CameraProjection() = default;
        CameraProjection(const CameraProjection& src) noexcept;
        CameraProjection& operator=(const CameraProjection& src) noexcept;

        bool UpdateMatrix(bool emitSignal = true) noexcept;

        constexpr void SetProjectionType(const ProjectionType projType) noexcept;
        constexpr void SetHeight(float height) noexcept;
        constexpr void SetAspectRatio(float ratio) noexcept;
        constexpr void SetAspectRatio(int width, int height) noexcept;
        constexpr void SetNear(float zNear) noexcept;
        constexpr void SetFar(float zFar) noexcept;

        constexpr void AdjustHeight(float delta) noexcept;
        constexpr void AdjustNear(float delta) noexcept;
        constexpr void AdjustFar(float delta) noexcept;

        constexpr ProjectionType GetProjectionType() const noexcept { return m_projType; }
        constexpr float GetHeight() const noexcept { return m_height; }
        constexpr float GetWidth() const noexcept { return (m_height * m_aspectRatio); }
        constexpr float GetAspectRatio() const noexcept { return m_aspectRatio; }
        constexpr float GetNear() const noexcept { return m_zNear; }
        constexpr float GetFar() const noexcept { return m_zFar; }
        float GetFovy() const noexcept;

        constexpr const glm::mat4& getProjectionMatrix() const noexcept { return m_projMat; }
        constexpr const glm::mat4& getProjectionInvMatrix() const noexcept { return m_projInvMat; }

        constexpr void explicitDirty() noexcept { m_dirty = true; }

    private:
        ProjectionType m_projType = ProjectionType::NONE;

        float m_height = constant::camera::DEFAULT_HEIGHT;
        float m_aspectRatio = constant::camera::DEFAULT_ASPECT_RATIO;
        float m_zNear = constant::camera::DEFAULT_Z_NEAR;
        float m_zFar = constant::camera::DEFAULT_Z_FAR;

        bool m_dirty = true;
        glm::mat4 m_projMat{ 1. };
        glm::mat4 m_projInvMat{ 1. };

        void OnUpdateProjectionMatrix_Ortho() noexcept;
        void OnUpdateProjectionMatrix_Perspective() noexcept;
        std::function<void(CameraProjection&)> m_onUpdateProjectionMatrix;

    private:
        glm::mat4 m_captureMat{ 1. };

    public:
        void SetCaptureMatrix(const glm::vec2& translate = { 0, 0 }, const glm::vec2& scale = { 1, 1 }) noexcept;
    };

    constexpr void CameraProjection::SetProjectionType(const ProjectionType projType) noexcept
    {
        if (m_projType == projType)
            return;

        m_projType = projType;

        switch (projType)
        {
        case ProjectionType::ORTHO:
            m_onUpdateProjectionMatrix = &CameraProjection::OnUpdateProjectionMatrix_Ortho;
            break;

        case ProjectionType::PERSPECTIVE:
            m_onUpdateProjectionMatrix = &CameraProjection::OnUpdateProjectionMatrix_Perspective;
            break;

        default:
            break;
        }

        m_dirty = true;
    }

    constexpr void CameraProjection::SetHeight(float height) noexcept
    {
        m_height = glm::clamp(height, constant::camera::MIN_HEIGHT, constant::camera::MAX_HEIGHT);
        m_dirty = true;
    }

    constexpr void CameraProjection::SetAspectRatio(float ratio) noexcept
    {
        m_aspectRatio = ratio;
        m_dirty = true;
    }

    constexpr void CameraProjection::SetAspectRatio(int width, int height) noexcept
    {
        SetAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    }

    constexpr void CameraProjection::SetNear(float zNear) noexcept
    {
        m_zNear = glm::clamp(zNear, constant::camera::MIN_Z_NEAR, m_zFar - glm::epsilon<float>());
        m_dirty = true;
    }

    constexpr void CameraProjection::SetFar(float zFar) noexcept
    {
        m_zFar = glm::clamp(zFar, m_zNear + glm::epsilon<float>(), constant::camera::MAX_Z_FAR);
        m_dirty = true;
    }

    constexpr void CameraProjection::AdjustHeight(float delta) noexcept
    {
        SetHeight(m_height + delta);
    }

    constexpr void CameraProjection::AdjustNear(float delta) noexcept
    {
        SetNear(m_zNear + delta);
    }

    constexpr void CameraProjection::AdjustFar(float delta) noexcept
    {
        SetFar(m_zFar + delta);
    }
}

#include "HeaderPost.h"