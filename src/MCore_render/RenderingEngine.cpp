#include "stdafx.h"
#include "RenderingEngine.h"

#include "../MCore_util/VulkanCore.h"
#include "Scene.h"

#include "RenderTargetFactory.h"
#include "RenderPassFactory.h"

#include "CameraUniform.h"

#include "PrefixRenderPass.h"
#include "OpaqueRenderPass.h"
#include "TransparentRenderPass.h"
#include "IDRenderPass.h"
#include "PostRenderPass.h"

using namespace mcore::render;

RenderingEngine::RenderingEngine(IScene* pScene)
	: IRenderingEngine{ pScene }
{}

void RenderingEngine::Initialize()
{
	InitTarget();
	InitRenderPass();
}

void RenderingEngine::OnResize(unsigned width, unsigned height)
{
	m_windowResize = { width, height };
}

void RenderingEngine::Draw()
{
	if (!Resize())
		return;

	Update();

	if (!m_bDraw)
		return;

	const auto& pRenderContext = GetScene()->GetRenderContext();
	const auto surfaceTexture = pRenderContext->GetNextSurfaceTexture();
	if (!surfaceTexture) return;

	{
		{
			auto commandID = pRenderContext->GetCommander()->CreateCommandBuffer(mvk::QueueType::GRAPHIC, mvk::CommandType::DYNAMIC);
			const auto commandBuffer = pRenderContext->GetCommander()->Get(commandID);
			
			vk::CommandBufferBeginInfo info{};
			commandBuffer.begin(info);

			GetRenderPass<PrefixRenderPass>()->Draw(commandBuffer);
			GetRenderPass<OpaqueRenderPass>()->Draw(commandBuffer);
			GetRenderPass<TransparentRenderPass>()->Draw(commandBuffer);
			GetRenderPass<PostRenderPass>()->MSAAToSurface(commandBuffer, surfaceTexture.get());

			commandBuffer.end();
			pRenderContext->GetCommandQueue()->AddCommand(std::move(commandID));
		}
		{
			auto commandID = pRenderContext->GetCommander()->CreateCommandBuffer(mvk::QueueType::GRAPHIC, mvk::CommandType::DYNAMIC);
			const auto commandBuffer = pRenderContext->GetCommander()->Get(commandID);

			vk::CommandBufferBeginInfo info{};
			commandBuffer.begin(info);

			GetRenderPass<IDRenderPass>()->Draw(commandBuffer);

			commandBuffer.end();
			pRenderContext->GetCommandQueue()->AddCommand(std::move(commandID));
		}

		pRenderContext->GetCommandQueue()->FlushAsync(vk::PipelineStageFlagBits2::eColorAttachmentOutput);
	}
	pRenderContext->Present();

	m_bDraw = false;
	return;
}

void RenderingEngine::Invalidate()
{
	m_bDraw = true;
}

void RenderingEngine::InitTarget()
{
	m_targets = RenderTargetFactory::Instance().Build(this);
	for (const auto& pTarget : GetTargets())
	{
		pTarget->Initialize();
	}
}

bool RenderingEngine::Resize()
{
	if (!m_windowResize) return true;

	const auto size = GetScene()->GetRenderContext()->GetSurfaceExtent();
	if (size.width == 0 || size.height == 0)
		return false;

	GetScene()->GetRenderContext()->Resize();
	GetScene()->UpdateViewport();

	const auto viewport = GetScene()->GetViewport();
	const auto aspectRatio = static_cast<float>(viewport.z) / static_cast<float>(viewport.w);
	auto& camera = GetScene()->GetUniform<CameraUniform>()->GetCamera();
	if (std::abs(camera.GetProjection().GetAspectRatio() - aspectRatio) > 1.0e-8f)
	{
		camera.GetProjection().SetAspectRatio(aspectRatio);
		camera.UpdateMatrix();
	}

	for (const auto& pTarget : GetTargets())
	{
		pTarget->Resize(viewport.z, viewport.w);
	}

	m_windowResize.reset();
	Invalidate();

	return true;
}

void RenderingEngine::InitRenderPass()
{
	m_renderPasses = RenderPassFactory::Instance().Build(this);
	for (auto& renderPass : m_renderPasses)
	{
		renderPass->Initialize();
	}
}

void RenderingEngine::Update()
{
	const auto& pRenderContext = GetScene()->GetRenderContext();
	const auto& pCommander = pRenderContext->GetCommander();

	auto commandID = pCommander->CreateCommandBuffer(mvk::QueueType::GRAPHIC, mvk::CommandType::DYNAMIC);
	const auto commandBuffer = pCommander->Get(commandID);

	vk::CommandBufferBeginInfo info{};
	commandBuffer.begin(info);

	bool bUniformUpdated = GetScene()->UpdateUniform(commandBuffer);
	bool bAABBUpdated = GetScene()->UpdateAABB(commandBuffer);

	commandBuffer.end();

	pRenderContext->GetCommandQueue()->AddCommand(commandID);
	pRenderContext->GetCommandQueue()->Flush(vk::PipelineStageFlagBits2::eColorAttachmentOutput).Wait();

	if (bUniformUpdated || bAABBUpdated)
	{
		Invalidate();
	}
}