#pragma once

#include "RenderModelDef.h"
#include "RenderConfigUniform.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ MeshModel : public IRenderModel
	{
		DECLARE_RENDER_MODEL(MeshModel)

	public:
		MeshModel(IRenderModelContainer* pModelContainer);

		void Initialize(const vk::CommandBuffer& commandBuffer) override;
		void PreProcess(IScene* pScene, const vk::CommandBuffer& commandBuffer) override;

	private:
		mvk::StagedBuffer<RenderConfigHostData> m_renderConfigBuffer;
	};
}

#include "HeaderPost.h"