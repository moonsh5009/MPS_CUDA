#pragma once

#include "RenderPassDef.h"

#include "IDRenderer.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ IDRenderPass : public IRenderPass
	{
		DECLARE_RENDER_PASS(IDRenderPass)

	public:
		IDRenderPass(IRenderingEngine* pRenderingEngine);

		void Initialize() override;
		void Draw(const vk::CommandBuffer& commandBuffer);

		void DrawID(const vk::CommandBuffer& commandBuffer);
		void Download(const vk::CommandBuffer& commandBuffer);

	private:
		IDRenderer m_idRenderer;
	};
}

#include "HeaderPost.h"