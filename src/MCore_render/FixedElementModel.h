#pragma once

#include "RenderModelDef.h"
#include "RenderConfigUniform.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ FixedElementModel : public IRenderModel
	{
		DECLARE_RENDER_MODEL(FixedElementModel)

	public:
		FixedElementModel(IRenderModelContainer* pModelContainer);

		void Initialize(const vk::CommandBuffer& commandBuffer) override;
		void PreProcess(IScene* pScene, const vk::CommandBuffer& commandBuffer) override;

	private:
		mvk::StagedBuffer<RenderConfigHostData> m_renderConfigBuffer;
	};
}

#include "HeaderPost.h"