#pragma once

#include "RenderPassDef.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ PostRenderPass : public IRenderPass
	{
		DECLARE_RENDER_PASS(PostRenderPass)

	public:
		PostRenderPass(IRenderingEngine* pRenderingEngine);

		void Initialize() override;
		void MSAAToSurface(const vk::CommandBuffer& commandBuffer, mvk::SurfaceTexture* surfaceTexture);
	};
}

#include "HeaderPost.h"