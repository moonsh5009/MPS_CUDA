#include "stdafx.h"
#include "RendererBase.h"

#include "../MCore_interface/IRenderModel.h"

using namespace mcore::render;

RendererBase::RendererBase(mcore::IRenderingEngine* pRenderingEngine, const std::string& label)
	: m_pRenderingEngine{ pRenderingEngine }
	, m_label{ label }
{}

void RendererBase::DrawPoint(
    const vk::CommandBuffer& commandBuffer,
    mvk::RenderContext* pRenderContext,
    IRenderModel* pModel)
{
    const auto& pointStream = pModel->GetPointStream();
    if (pointStream.IsEmpty())
        return;

	const auto& pipeline = LoadPointPipeline();
    if (!pipeline)
        return;

    pointStream.Draw(pipeline, commandBuffer, pRenderContext, pModel->GetBindGroupInstance());
}

void RendererBase::DrawLine(
    const vk::CommandBuffer& commandBuffer,
    mvk::RenderContext* pRenderContext,
    IRenderModel* pModel)
{
    const auto& lineStream = pModel->GetLineStream();
    if (lineStream.IsEmpty())
        return;

    const auto& pipeline = LoadLinePipeline();
    if (!pipeline)
        return;

    lineStream.Draw(pipeline, commandBuffer, pRenderContext, pModel->GetBindGroupInstance());
}

void RendererBase::DrawTriangle(
    const vk::CommandBuffer& commandBuffer,
    mvk::RenderContext* pRenderContext,
    IRenderModel* pModel)
{
    const auto& triangleStream = pModel->GetTriangleStream();
    if (triangleStream.IsEmpty())
        return;

    const auto& pipeline = LoadTrianglePipeline();
    if (!pipeline)
        return;

    triangleStream.Draw(pipeline, commandBuffer, pRenderContext, pModel->GetBindGroupInstance());
}
