#pragma once

#include "RenderPassDef.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ PrefixRenderPass : public IRenderPass
	{
		DECLARE_RENDER_PASS(PrefixRenderPass)

	public:
		PrefixRenderPass(IRenderingEngine* pRenderingEngine);

		void Initialize() override;
		void Draw(const vk::CommandBuffer& commandBuffer);

	private:
		void ResetColor(const vk::CommandBuffer& commandBuffer);
		void ResetID(const vk::CommandBuffer& commandBuffer);
	};
}

#include "HeaderPost.h"