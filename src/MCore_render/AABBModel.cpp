#include "stdafx.h"
#include "AABBModel.h"

#include "../MCore_interface/IRenderModelContainer.h"
#include "../MCore_interface/IScene.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_MODEL(AABBModel, RenderModelType::AABB)

AABBModel::AABBModel(IRenderModelContainer* pModelContainer)
	: IRenderModel{ pModelContainer }
	, m_renderConfigBuffer{ vk::BufferUsageFlagBits::eTransferDst }
{
	SetShow(false);
}

void AABBModel::Initialize(const vk::CommandBuffer& commandBuffer)
{
	m_aabb.Initialize();
	m_aabb += glm::vec3{ -0.5f };
	m_aabb += glm::vec3{ 0.5f };

	RenderConfigHostData hostData{};
	hostData.const_color = { 0.f, 0.f, 1.f, 1.f };
	hostData.color_type = ColorType::Const;
	hostData.transparent_type = ColorType::Const;
	hostData.model_type = static_cast<uint32_t>(RenderModelType::AABB);
	hostData.use_point_size = 1;
	hostData.point_size = 5.f;
	hostData.use_line_width = 1;
	hostData.line_width = 2.f;

	m_renderConfigBuffer.Resize(1);
	m_renderConfigBuffer.SetElement(0, hostData);
	m_renderConfigBuffer.Upload(commandBuffer);
}

void AABBModel::PreProcess(IScene* pScene, const vk::CommandBuffer& commandBuffer)
{
	const auto pRenderConfigUniform = pScene->GetUniform<RenderConfigUniform>();
	pRenderConfigUniform->CopyFrom(commandBuffer, m_renderConfigBuffer);
}