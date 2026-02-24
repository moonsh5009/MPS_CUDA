#include "stdafx.h"
#include "PrefixRenderPass.h"

#include "../MCore_interface/IScene.h"

#include "MSAARenderTarget.h"
#include "IDRenderTarget.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_PASS(PrefixRenderPass, RenderPassType::Prefix)

PrefixRenderPass::PrefixRenderPass(IRenderingEngine* pRenderingEngine)
	: IRenderPass{ pRenderingEngine }
{}

void PrefixRenderPass::Initialize()
{}

void PrefixRenderPass::Draw(const vk::CommandBuffer& commandBuffer)
{
	ResetColor(commandBuffer);
	ResetID(commandBuffer);
}

void PrefixRenderPass::ResetColor(const vk::CommandBuffer& commandBuffer)
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pMSAARenderTarget = GetRenderingEngine()->GetTarget<MSAARenderTarget>();
	const auto backgroundColor = pScene->GetBackgroundColor();

	vk::RenderingAttachmentInfo colorAttachment{};
	colorAttachment.imageView = pMSAARenderTarget->GetColorImage().GetView();
	colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	colorAttachment.clearValue = vk::ClearColorValue{ backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a };

	vk::RenderingAttachmentInfo depthAttachment{};
	depthAttachment.imageView = pMSAARenderTarget->GetDepthImage().GetView();
	depthAttachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
	depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	depthAttachment.clearValue = vk::ClearDepthStencilValue{ 0.0f, 0 };

	vk::RenderingInfo renderingInfo{};
	renderingInfo.renderArea = pScene->GetRenderContext()->GetRenderArea();
	renderingInfo.layerCount = 1;
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.endRendering();
}

void mcore::render::PrefixRenderPass::ResetID(const vk::CommandBuffer& commandBuffer)
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto pIDRenderTarget = GetRenderingEngine()->GetTarget<IDRenderTarget>();

	vk::RenderingAttachmentInfo colorAttachment{};
	colorAttachment.imageView = pIDRenderTarget->GetIDTexture().GetView();
	colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	colorAttachment.clearValue = vk::ClearColorValue{ 0, 0, 0, 0 };

	vk::RenderingAttachmentInfo depthAttachment{};
	depthAttachment.imageView = pIDRenderTarget->GetDepthTexture().GetView();
	depthAttachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
	depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	depthAttachment.clearValue = vk::ClearDepthStencilValue{ 0.0f, 0 };

	vk::RenderingInfo renderingInfo{};
	renderingInfo.renderArea = pScene->GetRenderContext()->GetRenderArea();
	renderingInfo.layerCount = 1;
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.endRendering();
	
}
