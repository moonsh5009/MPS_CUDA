#pragma once

#include "../MCore_interface/IUserInputHandler.h"

#include "CameraUserInputHandler.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ UserInputHandler : public IUserInputHandler
    {
    public:
        UserInputHandler(IScene* pScene) noexcept;

        void OnMouseUp(MouseEventInfo info) override;
        void OnMouseDown(MouseEventInfo info) override;
        void OnMouseMove(MouseEventInfo info) override;
        bool OnMouseWheel(MouseEventInfo info) override;
        void OnKeyUp(KeyEventInfo info) override;
        void OnKeyDown(KeyEventInfo info) override;
        void OnResize(unsigned width, unsigned height) override;
        void OnDraw() override;

    private:
        std::unique_ptr<CameraUserInputHandler> m_pCameraUserInputHandler;
    };
}

#include "HeaderPost.h"