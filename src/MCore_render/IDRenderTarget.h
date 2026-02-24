#pragma once

#include "../MCore_util/Texture.h"

#include "RenderTargetDef.h"

#include "HeaderPre.h"

namespace mcore::render
{
    class __MY_EXT_CLASS__ IDRenderTarget : public IRenderTarget
    {
        DECLARE_RENDER_TARGET(IDRenderTarget)

    public:
        IDRenderTarget(IRenderingEngine* pScene);

        void Initialize() override;
        void Resize(unsigned width, unsigned height) override;

        std::tuple<uint32_t, uint32_t, uint32_t> GetID(uint32_t x, uint32_t y);
        float GetDepth(uint32_t x, uint32_t y);

        mvk::Texture& GetIDTexture() { return m_idTexture; }
        mvk::Texture& GetDepthTexture() { return m_depthTexture; }
        const mvk::Texture& GetIDTexture() const { return m_idTexture; }
        const mvk::Texture& GetDepthTexture() const { return m_depthTexture; }

    private:
        mvk::Texture m_idTexture;
        mvk::Texture m_depthTexture;
    };
}

#include "HeaderPost.h"