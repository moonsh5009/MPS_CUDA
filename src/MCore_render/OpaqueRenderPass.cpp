#include "stdafx.h"
#include "OpaqueRenderPass.h"

#include "../MCore_interface/IRenderCore.h"

#include "MSAARenderTarget.h"

#include "MeshModel.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_PASS(OpaqueRenderPass, RenderPassType::Opaque)

OpaqueRenderPass::OpaqueRenderPass(IRenderingEngine* pRenderingEngine)
	: IRenderPass{ pRenderingEngine }
	, m_opaqueRenderer{ pRenderingEngine }
{}

void OpaqueRenderPass::Initialize()
{
}

void OpaqueRenderPass::Draw(const vk::CommandBuffer& commandBuffer)
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pRenderCore = pScene->GetRenderCore();
	const auto pModelContainer = pRenderCore->GetModelContainer();

	const auto pMSAARenderTarget = GetRenderingEngine()->GetTarget<MSAARenderTarget>();

	vk::RenderingAttachmentInfo colorAttachment{};
	colorAttachment.imageView = pMSAARenderTarget->GetColorImage().GetView();
	colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;

	vk::RenderingAttachmentInfo depthAttachment{};
	depthAttachment.imageView = pMSAARenderTarget->GetDepthImage().GetView();
	depthAttachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
	depthAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
	depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;

	vk::RenderingInfo renderingInfo{};
	renderingInfo.renderArea = pScene->GetRenderContext()->GetRenderArea();
	renderingInfo.layerCount = 1;
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	for (const auto& pModel : pModelContainer->GetModels())
	{
		if (!pModel || !pModel->GetShow()) continue;

		pModel->PreProcess(pScene, commandBuffer);

		commandBuffer.beginRendering(renderingInfo);
		m_opaqueRenderer.DrawTriangle(commandBuffer, pScene->GetRenderContext().get(), pModel.get());
		m_opaqueRenderer.DrawLine(commandBuffer, pScene->GetRenderContext().get(), pModel.get());
		m_opaqueRenderer.DrawPoint(commandBuffer, pScene->GetRenderContext().get(), pModel.get());
		commandBuffer.endRendering();
	}
}
