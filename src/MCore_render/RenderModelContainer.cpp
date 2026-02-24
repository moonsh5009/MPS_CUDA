#include "stdafx.h"
#include "RenderModelContainer.h"

#include "../MCore_util/VulkanCore.h"

#include "../MCore_interface/IRenderingEngine.h"

#include "CameraUniform.h"

#include "RenderUniformFactory.h"
#include "RenderTargetFactory.h"
#include "RenderModelFactory.h"

using namespace mcore::render;

RenderModelContainer::RenderModelContainer(IRenderCore* pRenderCore)
	: IRenderModelContainer{ pRenderCore }
{}

void RenderModelContainer::Initialize()
{
	auto cmdBuffer = mvk::VulkanCore::Instance()->CreateSimpleCommandBuffer(mvk::QueueType::TRANSFER);
	InitModel(cmdBuffer);
	mvk::VulkanCore::Instance()->SimpleSubmit(std::move(cmdBuffer));
}

void RenderModelContainer::InitModel(const vk::CommandBuffer& commandBuffer)
{
	m_models = RenderModelFactory::Instance().Build(this);
	for (const auto& pModel : GetModels())
	{
		if (pModel)
			pModel->Initialize(commandBuffer);
	}
}