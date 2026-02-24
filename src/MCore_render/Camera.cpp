#include "stdafx.h"
#include "Camera.h"

namespace mcore::render
{
    void Camera::Initialize()
    {
        m_updateListener.Bind(m_transform.onUpdateMatrix, [&]()
        {
            if (m_ignoreUpdateFlag)
                return;

            onUpdateMatrix->Dispatch();
        });

        m_updateListener.Bind(m_projection.onUpdateMatrix, [&]()
        {
            if (m_ignoreUpdateFlag)
                return;

            onUpdateMatrix->Dispatch();
        });
    }

    bool Camera::UpdateMatrix(bool emitSignal)
    {
        bool needToUpdate = false;

        m_ignoreUpdateFlag = true;
        needToUpdate |= m_transform.UpdateMatrix(emitSignal);
        needToUpdate |= m_projection.UpdateMatrix(emitSignal);
        m_ignoreUpdateFlag = false;

        if (!needToUpdate)
            return false;

        if (emitSignal)
            onUpdateMatrix->Dispatch();

        return true;
    }
}