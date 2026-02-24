#pragma once

#include "RenderPassDef.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ TransparentRenderPass : public IRenderPass
	{
		DECLARE_RENDER_PASS(TransparentRenderPass)

	public:
		TransparentRenderPass(IRenderingEngine* pRenderingEngine);

		void Initialize() override;
		void Draw(const vk::CommandBuffer& commandBuffer);
	};
}

#include "HeaderPost.h"