#include "stdafx.h"
#include "TransparentRenderPass.h"

#include "../MCore_interface/IRenderingEngine.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_PASS(TransparentRenderPass, RenderPassType::Transparent)

TransparentRenderPass::TransparentRenderPass(IRenderingEngine* pRenderingEngine)
	: IRenderPass{ pRenderingEngine }
{}

void TransparentRenderPass::Initialize()
{}

void TransparentRenderPass::Draw(const vk::CommandBuffer& commandBuffer)
{}
