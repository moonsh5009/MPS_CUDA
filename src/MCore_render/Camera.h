#pragma once

#include "../MCore_util/SignalSlot.h"

#include "CameraTransform.h"
#include "CameraProjection.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ Camera
    {
    public:
        mcore::Signal<void()> onUpdateMatrix = mcore::MakeSignal<void()>();

        void Initialize();
        bool UpdateMatrix(bool emitSignal = true);

        constexpr CameraTransform& GetTransform() noexcept { return m_transform; }
        constexpr const CameraTransform& GetTransform() const noexcept { return m_transform; }

        constexpr CameraProjection& GetProjection() noexcept { return m_projection; }
        constexpr const CameraProjection& GetProjection() const noexcept { return m_projection; }

    private:
        CameraTransform m_transform;
        CameraProjection m_projection;

        bool m_ignoreUpdateFlag = false;
        mcore::SignalSlot m_updateListener;
    };
}

#include "HeaderPost.h"
