#pragma once

#include "RendererBase.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class __MY_EXT_CLASS__ IDRenderer : public RendererBase
	{
	public:
		IDRenderer(IRenderingEngine* pRenderingEngine);

	protected:
		const mvk::GraphicsPipeline& LoadPointPipeline() override;
		const mvk::GraphicsPipeline& LoadLinePipeline() override;
		const mvk::GraphicsPipeline& LoadTrianglePipeline() override;

		mvk::GraphicsPipeline m_pointPipeline;
		mvk::GraphicsPipeline m_linePipeline;
		mvk::GraphicsPipeline m_trianglePipeline;
	};
}

#include "HeaderPost.h"