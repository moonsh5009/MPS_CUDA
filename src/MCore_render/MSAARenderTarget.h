#pragma once

#include "../MCore_util/Texture.h"

#include "RenderTargetDef.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ MSAARenderTarget : public IRenderTarget
    {
        DECLARE_RENDER_TARGET(MSAARenderTarget)

    public:
        MSAARenderTarget(IRenderingEngine* pScene);

        void Initialize() override;
        void Resize(unsigned width, unsigned height) override;

        const mvk::Texture& GetColorImage() const { return m_colorTexture; }
        const mvk::Texture& GetDepthImage() const { return m_depthTexture; }

    private:
        mvk::Texture m_colorTexture;
        mvk::Texture m_depthTexture;
    };
}

#include "HeaderPost.h"