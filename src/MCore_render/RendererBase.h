#pragma once

#include "../MCore_interface/StreamBuffer.h"

#include "HeaderPre.h"

namespace mcore
{
	class IRenderingEngine;
	class IRenderModel;
}
namespace mcore::render
{
	class __MY_EXT_CLASS__ RendererBase
	{
	public:
		RendererBase(mcore::IRenderingEngine* pRenderingEngine, const std::string& label);
		virtual ~RendererBase() = default;
		RendererBase(const RendererBase&) = delete;
		RendererBase(RendererBase&&) noexcept = default;
		RendererBase& operator=(const RendererBase&) = delete;
		RendererBase& operator=(RendererBase&&) noexcept = default;

		void DrawPoint(const vk::CommandBuffer& commandBuffer,
			mvk::RenderContext* pRenderContext,
			IRenderModel* pModel);
		void DrawLine(const vk::CommandBuffer& commandBuffer,
			mvk::RenderContext* pRenderContext,
			IRenderModel* pModel);
		void DrawTriangle(const vk::CommandBuffer& commandBuffer,
			mvk::RenderContext* pRenderContext,
			IRenderModel* pModel);

		const std::string& GetLabel() const { return m_label; }

	protected:
		mcore::IRenderingEngine* GetRenderingEngine() const { return m_pRenderingEngine; }

		virtual const mvk::GraphicsPipeline& LoadPointPipeline() = 0;
		virtual const mvk::GraphicsPipeline& LoadLinePipeline() = 0;
		virtual const mvk::GraphicsPipeline& LoadTrianglePipeline() = 0;

	private:
		mcore::IRenderingEngine* m_pRenderingEngine;
		std::string m_label;
	};
}

#include "HeaderPost.h"