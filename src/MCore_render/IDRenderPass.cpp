#include "stdafx.h"
#include "IDRenderPass.h"

#include "../MCore_interface/IRenderCore.h"

#include "RenderConfigUniform.h"

#include "IDRenderTarget.h"

#include "MeshModel.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_PASS(IDRenderPass, RenderPassType::ID)

IDRenderPass::IDRenderPass(IRenderingEngine* pRenderingEngine)
	: IRenderPass{ pRenderingEngine }
	, m_idRenderer{ pRenderingEngine }
{}

void IDRenderPass::Initialize()
{}

void IDRenderPass::Draw(const vk::CommandBuffer& commandBuffer)
{
	DrawID(commandBuffer);
	Download(commandBuffer);
}

void IDRenderPass::DrawID(const vk::CommandBuffer& commandBuffer)
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pRenderCore = pScene->GetRenderCore();
	const auto pModelContainer = pRenderCore->GetModelContainer();

	const auto pRenderConfigUniform = pScene->GetUniform<RenderConfigUniform>();

	const auto pIDRenderTarget = GetRenderingEngine()->GetTarget<IDRenderTarget>();

	vk::RenderingAttachmentInfo colorAttachment{};
	colorAttachment.imageView = pIDRenderTarget->GetIDTexture().GetView();
	colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;

	vk::RenderingAttachmentInfo depthAttachment{};
	depthAttachment.imageView = pIDRenderTarget->GetDepthTexture().GetView();
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

		m_idRenderer.DrawTriangle(commandBuffer, pScene->GetRenderContext().get(), pModel.get());
		m_idRenderer.DrawLine(commandBuffer, pScene->GetRenderContext().get(), pModel.get());
		m_idRenderer.DrawPoint(commandBuffer, pScene->GetRenderContext().get(), pModel.get());

		commandBuffer.endRendering();
	}
}

void IDRenderPass::Download(const vk::CommandBuffer& commandBuffer)
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pIDRenderTarget = GetRenderingEngine()->GetTarget<IDRenderTarget>();

	pIDRenderTarget->GetIDTexture().TransitionLayout(commandBuffer, vk::ImageLayout::eTransferSrcOptimal);
	pIDRenderTarget->GetDepthTexture().TransitionLayout(commandBuffer, vk::ImageLayout::eTransferSrcOptimal);

	pIDRenderTarget->GetIDTexture().Download(commandBuffer);
	pIDRenderTarget->GetDepthTexture().Download(commandBuffer);

	pIDRenderTarget->GetIDTexture().TransitionLayout(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal);
	pIDRenderTarget->GetDepthTexture().TransitionLayout(commandBuffer, vk::ImageLayout::eDepthAttachmentOptimal);
}