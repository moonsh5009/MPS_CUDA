#pragma once

#include "../MCore_util/WindowEventInfo.h"
#include <memory>

namespace mcore
{
    class IScene;
    class IUserInputHandler
    {
    public:
        IUserInputHandler(IScene* pScene) : m_pScene{ pScene } {}
        virtual ~IUserInputHandler() = default;
        IUserInputHandler(const IUserInputHandler&) = delete;
        IUserInputHandler(IUserInputHandler&&) noexcept = default;
        IUserInputHandler& operator=(const IUserInputHandler&) = delete;
        IUserInputHandler& operator=(IUserInputHandler&&) noexcept = default;

        virtual void OnMouseUp(MouseEventInfo info) = 0;
        virtual void OnMouseDown(MouseEventInfo info) = 0;
        virtual void OnMouseMove(MouseEventInfo info) = 0;
        virtual bool OnMouseWheel(MouseEventInfo info) = 0;
        virtual void OnKeyUp(KeyEventInfo info) = 0;
        virtual void OnKeyDown(KeyEventInfo info) = 0;
        virtual void OnResize(unsigned width, unsigned height) = 0;
        virtual void OnDraw() = 0;

        IScene* GetScene() const { return m_pScene; }

    private:
        IScene* m_pScene;
    };
}