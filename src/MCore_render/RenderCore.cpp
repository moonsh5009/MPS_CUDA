#include "stdafx.h"
#include "RenderCore.h"

#include "RenderModelContainer.h"
#include "Scene.h"

#include "RenderUniformFactory.h"

#include <ranges>

using namespace mcore::render;

void RenderCore::Initialize()
{
	m_pModelContainer = std::make_unique<RenderModelContainer>(this);
	m_pModelContainer->Initialize();

	m_uniformBindGroupLayout = RenderUniformFactory::Instance().BuildBindGroupLayout();
}

void RenderCore::AddScene(HWND window)
{
	auto pRenderContext = std::make_shared<mvk::RenderContext>();
	pRenderContext->Initialize(window);

	auto pScene = std::make_unique<Scene>(this);
	pScene->Initialize(std::move(pRenderContext));
	pScene->SetCameraMode(CameraDirectionType::PERSPECTIVE, true);
	m_scenes.emplace(window, std::move(pScene));
}

void RenderCore::Run()
{}

void RenderCore::Invalidate()
{
	for (auto& pScene : m_scenes | std::views::values)
	{
		pScene->Invalidate();
	}
}

void RenderCore::SetZoomFitAllScenes()
{
	for (auto& pScene : m_scenes | std::views::values)
	{
		pScene->SetZoomFit();
	}
}

vk::Format RenderCore::GetColorFormat() const
{
	return vk::Format::eB8G8R8A8Srgb;
}

vk::Format RenderCore::GetDepthFormat() const
{
	return vk::Format::eD32Sfloat;
}

vk::SampleCountFlagBits RenderCore::GetMultiSampleCount() const
{
	return vk::SampleCountFlagBits::e4;
}

vk::PipelineMultisampleStateCreateInfo RenderCore::GetMultiSampleState() const
{
	vk::PipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = GetMultiSampleCount();
	multisampling.minSampleShading = 1.0f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;
	return multisampling;
}

const mvk::BindGroupLayout& RenderCore::GetUniformBindGrouplayout() const
{
	return m_uniformBindGroupLayout;
}

mvk::BindGroupLayout RenderCore::GetPointStreamBindGrouplayout() const
{
	constexpr auto totalInstances = static_cast<uint32_t>(enum_size<RenderModelType>()) * mvk::MAX_FRAMES_IN_FLIGHT;

	return mvk::BindGroupLayoutBuilder()
		.AddStorageBuffer(0, vk::ShaderStageFlagBits::eVertex)
		.AddStorageBuffer(1, vk::ShaderStageFlagBits::eVertex)
		.AddStorageBuffer(2, vk::ShaderStageFlagBits::eVertex)
		.AddStorageBuffer(3, vk::ShaderStageFlagBits::eVertex)
		.Build(totalInstances);
}

mvk::BindGroupLayout RenderCore::GetLineStreamBindGrouplayout() const
{
	constexpr auto totalInstances = static_cast<uint32_t>(enum_size<RenderModelType>()) * mvk::MAX_FRAMES_IN_FLIGHT;
	return mvk::BindGroupLayoutBuilder()
		.AddStorageBuffer(0, vk::ShaderStageFlagBits::eVertex)
		.AddStorageBuffer(1, vk::ShaderStageFlagBits::eVertex)
		.AddStorageBuffer(2, vk::ShaderStageFlagBits::eVertex)
		.AddStorageBuffer(3, vk::ShaderStageFlagBits::eVertex)
		.Build(totalInstances);
}
