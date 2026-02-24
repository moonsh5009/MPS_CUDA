#include "stdafx.h"
#include "MSAARenderTarget.h"

#include "../MCore_interface/IScene.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_TARGET(MSAARenderTarget, RenderTargetType::MSAA)

MSAARenderTarget::MSAARenderTarget(IRenderingEngine* pRenderingEngine)
	: IRenderTarget{ pRenderingEngine }
{}

void MSAARenderTarget::Initialize()
{
	const auto pScene = GetRenderingEngine()->GetScene();
	const auto format = pScene->GetColorFormat();
	m_colorTexture.Create(
		format,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::SampleCountFlagBits::e4);
	m_depthTexture.Create(
		pScene->GetDepthFormat(),
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::SampleCountFlagBits::e4);
}

void MSAARenderTarget::Resize(unsigned width, unsigned height)
{
	vk::Extent3D extent{ width, height, 1 };
	m_colorTexture.Resize(extent);
	m_depthTexture.Resize(extent);
}
