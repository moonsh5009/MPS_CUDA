#pragma once

#include "../MCore_util/StagedBuffer.h"
#include "../MCore_util/AABB.h"

#include "RenderModelDef.h"
#include "RenderConfigUniform.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ AABBModel : public IRenderModel
	{
		DECLARE_RENDER_MODEL(AABBModel)

	public:
		AABBModel(IRenderModelContainer* pModelContainer);

		void Initialize(const vk::CommandBuffer& commandBuffer) override;
		void PreProcess(IScene* pScene, const vk::CommandBuffer& commandBuffer) override;

		void SetAABB(const AABBf& aabb) { m_aabb = aabb; }
		const AABBf& GetAABB() const { return m_aabb; }

	private:
		mvk::StagedBuffer<RenderConfigHostData> m_renderConfigBuffer;
		AABBf m_aabb;
	};
}

#include "HeaderPost.h"