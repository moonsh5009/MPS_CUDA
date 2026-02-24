#pragma once

#include "../MCore_util/WindowEventInfo.h"

#include "HeaderPre.h"

namespace mcore
{
    class IScene;
    namespace render
    {
        class __MY_EXT_CLASS__ CameraUserInputHandler
        {
        public:
            CameraUserInputHandler(IScene* pScene) noexcept;

            void OnMouseUp(MouseEventInfo info);
            void OnMouseDown(MouseEventInfo info);
            void OnMouseMove(MouseEventInfo info);
            bool OnMouseWheel(MouseEventInfo info);
            void OnKeyUp(KeyEventInfo info);
            void OnKeyDown(KeyEventInfo info);

            bool IsCameraMoving() const { return m_bMoving || m_bRotating; }
            glm::ivec2 GetMouseViewportPos() const noexcept;

            IScene* GetScene() const { return m_pScene; }

        private:
            glm::ivec2 ConvertMousePosToScreenPos(const std::pair<int, int>& mousePos) const noexcept;
            void UpdateMovePivot() noexcept;

            void Translate(const std::pair<int, int>& mousePoint);
            void Rotate(const std::pair<int, int>& mousePoint);
            void Zoom(float delta, const std::pair<int, int>& mousePoint);
            void Advance(float delta, const std::pair<int, int>& mousePoint);

            IScene* m_pScene;

            bool m_bMoving = false;
            bool m_bRotating = false;
            bool m_bScroll = false;
            float m_depthMovePivot = 0.f;
            glm::vec3 m_posMovePivot = { 0.f, 0.f, 0.f };
            std::pair<int, int> m_prevMousePoint = { 0, 0 };
        };
    }
}

#include "HeaderPost.h"