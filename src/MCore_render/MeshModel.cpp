#include "stdafx.h"
#include "MeshModel.h"

#include "../MCore_interface/IRenderModelContainer.h"
#include "../MCore_interface/IScene.h"

using namespace mcore;
using namespace mcore::render;

IMPLEMENT_RENDER_MODEL(MeshModel, RenderModelType::Mesh)

MeshModel::MeshModel(IRenderModelContainer* pModelContainer)
	: IRenderModel{ pModelContainer }
	, m_renderConfigBuffer{ vk::BufferUsageFlagBits::eTransferDst }
{}

void MeshModel::Initialize(const vk::CommandBuffer& commandBuffer)
{
	RenderConfigHostData hostData{};
	hostData.color_type = ColorType::Attribute;
	hostData.transparent_type = ColorType::Attribute;
	hostData.model_type = static_cast<uint32_t>(RenderModelType::Mesh);
	hostData.use_point_size = 0;
	hostData.use_line_width = 0;

	m_renderConfigBuffer.Resize(1);
	m_renderConfigBuffer.SetElement(0, hostData);
	m_renderConfigBuffer.Upload(commandBuffer);
}

void MeshModel::PreProcess(IScene* pScene, const vk::CommandBuffer& commandBuffer)
{
	const auto pRenderConfigUniform = pScene->GetUniform<RenderConfigUniform>();
	pRenderConfigUniform->CopyFrom(commandBuffer, m_renderConfigBuffer);
}
