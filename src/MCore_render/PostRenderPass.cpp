#include "stdafx.h"
#include "PostRenderPass.h"

#include "../MCore_interface/IScene.h"

#include "MSAARenderTarget.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_PASS(PostRenderPass, RenderPassType::Post)

PostRenderPass::PostRenderPass(IRenderingEngine* pRenderingEngine)
	: IRenderPass{ pRenderingEngine }
{}

void PostRenderPass::Initialize()
{}

void PostRenderPass::MSAAToSurface(const vk::CommandBuffer& commandBuffer, mvk::SurfaceTexture* surfaceTexture)
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pMSAARenderTarget = GetRenderingEngine()->GetTarget<MSAARenderTarget>();

	vk::RenderingAttachmentInfo colorAttachment{};
	colorAttachment.imageView = pMSAARenderTarget->GetColorImage().GetView();
	colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.resolveImageView = surfaceTexture->GetView();
	colorAttachment.resolveMode = vk::ResolveModeFlagBits::eAverage;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;

	vk::RenderingInfo renderingInfo{};
	renderingInfo.renderArea = pScene->GetRenderContext()->GetRenderArea();
	renderingInfo.layerCount = 1;
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.endRendering();
}
