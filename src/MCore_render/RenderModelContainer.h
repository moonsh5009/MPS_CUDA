#pragma once

#include "../MCore_interface/IRenderModelContainer.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ RenderModelContainer : public IRenderModelContainer
	{
	public:
		RenderModelContainer(IRenderCore* pRenderCore);

		void Initialize() override;

	private:
		void InitModel(const vk::CommandBuffer& commandBuffer);
	};
}

#include "HeaderPost.h"