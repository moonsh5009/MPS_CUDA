#include "stdafx.h"
#include "IDRenderTarget.h"

#include "../MCore_interface/IScene.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_TARGET(IDRenderTarget, RenderTargetType::ID)

IDRenderTarget::IDRenderTarget(IRenderingEngine* pRenderingEngine)
	: IRenderTarget{ pRenderingEngine }
{}

void IDRenderTarget::Initialize()
{
	const auto pScene = GetRenderingEngine()->GetScene();
	m_idTexture.Create(
		RENDER_ID_TEXTURE_FORMAT,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::SampleCountFlagBits::e1);
	m_depthTexture.Create(
		pScene->GetDepthFormat(),
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::SampleCountFlagBits::e1);
}

void IDRenderTarget::Resize(unsigned width, unsigned height)
{
	vk::Extent3D extent{ width, height, 1 };
	m_idTexture.Resize(extent);
	m_depthTexture.Resize(extent);
}

std::tuple<uint32_t, uint32_t, uint32_t> IDRenderTarget::GetID(uint32_t x, uint32_t y)
{
	if (!m_idTexture.IsValid())
	{
		mcore::Logger::Warning("ID Texture is not valid");
		mcore::Logger::Print();
		return { 0, 0, 0 };
	}

	const auto extent = m_idTexture.GetExtent();
	if (x >= extent.width || y >= extent.height)
	{
		mcore::Logger::Warning("GetID: coordinates out of bounds (", x, ", ", y, ")");
		mcore::Logger::Print();
		return { 0, 0, 0 };
	}
	
	std::array<uint32_t, 4> pixelData;
	m_idTexture.CopyToHost(pixelData.data(), x, y);
	return { pixelData[0], pixelData[1], pixelData[2] };
}

float IDRenderTarget::GetDepth(uint32_t x, uint32_t y)
{
	if (!m_depthTexture.IsValid())
	{
		mcore::Logger::Warning("ID Texture is not valid");
		mcore::Logger::Print();
		return 0.f;
	}

	const auto extent = m_depthTexture.GetExtent();
	if (x >= extent.width || y >= extent.height)
	{
		mcore::Logger::Warning("GetID: coordinates out of bounds (", x, ", ", y, ")");
		mcore::Logger::Print();
		return 0.f;
	}

	float depth = 0.f;
	m_depthTexture.CopyToHost(&depth, x, y);
	return depth;
}
