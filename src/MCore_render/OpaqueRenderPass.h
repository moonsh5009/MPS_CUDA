#pragma once

#include "RenderPassDef.h"

#include "OpaqueRenderer.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ OpaqueRenderPass : public IRenderPass
	{
		DECLARE_RENDER_PASS(OpaqueRenderPass)

	public:
		OpaqueRenderPass(IRenderingEngine* pRenderingEngine);

		void Initialize() override;
		void Draw(const vk::CommandBuffer& commandBuffer);

	private:
		OpaqueRenderer m_opaqueRenderer;
	};
}

#include "HeaderPost.h"