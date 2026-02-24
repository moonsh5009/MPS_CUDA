#include "stdafx.h"
#include "Scene.h"

#include "../MCore_util/Commander.h"

#include "../MCore_interface/IRenderCore.h"

#include "RenderingEngine.h"
#include "UserInputHandler.h"

#include "RenderUniformFactory.h"

#include "CameraUniform.h"

#include "AABBModel.h"

using namespace mcore;
using namespace mcore::render;

Scene::Scene(IRenderCore* pRenderCore)
	: IScene{ pRenderCore }
	, m_viewport{ 0, 0, 0, 0 }
	, m_backgroundColor{ 0.9f, 0.9f, 0.9f, 1.f }
{
}

void Scene::Initialize(std::shared_ptr<mvk::RenderContext>&& pRenderContext)
{
	m_pRenderContext = std::move(pRenderContext);

	m_pRenderingEngine = std::make_unique<RenderingEngine>(this);
	m_pRenderingEngine->Initialize();

	m_pUserInputHandler = std::make_unique<UserInputHandler>(this);

	InitUniform();
	InitAABB();

	UpdateViewport();
	OnResize(m_viewport.z, m_viewport.w);
}

void Scene::OnResize(unsigned width, unsigned height)
{
	m_pRenderingEngine->OnResize(m_viewport.z, m_viewport.w);
}

void Scene::Draw()
{
	GetRenderingEngine()->Draw();
}

void Scene::Invalidate()
{
	GetRenderingEngine()->Invalidate();
}

void Scene::SetZoomFit()
{
	GetUniform<CameraUniform>()->SetZoomFit();
}

void Scene::UpdateViewport()
{
	m_viewport = {
		0, 0,
		GetRenderContext()->GetImageSize().width,
		GetRenderContext()->GetImageSize().height
	};
}

bool Scene::UpdateAABB(const vk::CommandBuffer& commandBuffer)
{
	/*if (m_aabb.IsAvailable())
	{
		return false;
	}*/

	if (m_aabb == GetModelContainer()->GetModel<AABBModel>()->GetAABB())
		return false;
	
	m_aabb = GetModelContainer()->GetModel<AABBModel>()->GetAABB();
	onUpdateAABB->Dispatch(m_aabb);
	return true;
}

bool Scene::UpdateUniform(const vk::CommandBuffer& commandBuffer)
{
	bool bUpdate = false;
	for (auto& pUniform : GetUniforms())
	{
		bUpdate |= pUniform->Update(commandBuffer);
	}
	return bUpdate;
}

void Scene::SetCameraMode(CameraDirectionType dirType, bool is3D) const
{
	const auto pCameraUniform = GetUniform<CameraUniform>();
	auto& camera = pCameraUniform->GetCamera();
	camera.GetProjection().SetProjectionType(is3D ? ProjectionType::PERSPECTIVE : ProjectionType::ORTHO);

	switch (dirType)
	{
	case CameraDirectionType::FRONT:
		camera.GetTransform().Orient({ -1., 0., 0. }, { 0., 0., 1. });
		break;
	case CameraDirectionType::RIGHT:
		camera.GetTransform().Orient({ 0., -1., 0. }, { 0., 0., 1. });
		break;
	case CameraDirectionType::BOTTOM:
		camera.GetTransform().Orient({ 0., 0., 1. }, { 1., 0., 0. });
		break;
	case CameraDirectionType::PERSPECTIVE:
		camera.GetTransform().Orient({ -0.6, -0.3, 0.3 }, { 0., 0., 1. });
		break;
	default:
		break;
	}
	camera.UpdateMatrix();
}

IRenderModelContainer* Scene::GetModelContainer() const
{
	return GetRenderCore()->GetModelContainer();
}

vk::Format Scene::GetSwapchainFormat() const
{
	return GetRenderContext()->GetFormat();
}

vk::Format Scene::GetColorFormat() const
{
	return GetRenderCore()->GetColorFormat();
}

vk::Format Scene::GetDepthFormat() const
{
	return GetRenderCore()->GetDepthFormat();
}

vk::SampleCountFlagBits Scene::GetMultiSampleCount() const
{
	return GetRenderCore()->GetMultiSampleCount();
}

vk::PipelineMultisampleStateCreateInfo Scene::GetMultiSampleState() const
{
	return GetRenderCore()->GetMultiSampleState();
}

void Scene::InitUniform()
{
	m_uniforms = RenderUniformFactory::Instance().Build(this);
	for (const auto& pUniform : GetUniforms())
	{
		pUniform->Initialize();
	}

	auto binder = GetRenderCore()->GetUniformBindGrouplayout()->Binder();
	for (const auto& pUniform : GetUniforms())
	{
		binder = std::move(pUniform->BindBufferToLayout(std::move(binder)));
	}
	std::move(binder).Commit();
}

void Scene::InitAABB()
{
	m_aabb.Initialize();
	m_aabb += { -1.f, -1.f, -1.f };
	m_aabb += { 1.f, 1.f, 1.f };
}