#include "stdafx.h"
#include "MPSRenderManager.h"

#include "../MPS_Object/MPSGBArchiver.h"

#include "../MPS_Object/MPSSPHModel.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSSPHObject.h"

#include "../MPS_Object/MPSSpatialHash.h"

#include "MPSSPHRenderer.h"

mps::rndr::RenderManager::RenderManager() : m_pGLPArchiver{ std::make_unique<GLPArchiver>() }
{
	m_pGLPArchiver->Initalize();

	m_mRndrID.reserve(RENDERER::Size);
	m_mRndrID.emplace(typeid(mps::SPHModel).hash_code(), RENDERER::SPH);

	m_aRenderer.resize(RENDERER::Size);
	m_aRenderer[RENDERER::SPH] = std::make_shared<mps::rndr::SPHRenderer>();

	m_aModel.resize(RENDERER::Size);
}

void mps::rndr::RenderManager::Initalize(const mps::GBArchiver* pGBArchiver)
{
	for (auto& pRenderer : m_aRenderer)
	{
		pRenderer->Initalize(m_pGLPArchiver.get(), pGBArchiver);
	}
}

void mps::rndr::RenderManager::AddModel(const std::shared_ptr<Model>& pModel)
{
	const auto rndrID = m_mRndrID[pModel->GetTypeID()];
	m_aModel[rndrID].emplace(pModel);
}

void mps::rndr::RenderManager::Draw(const mps::GBArchiver* pGBArchiver)
{
	for (uint32_t rndrID = 0; rndrID < RENDERER::Size; rndrID++)
	{
		for (const auto pModel : m_aModel[rndrID])
		{
			m_aRenderer[rndrID]->Draw(m_pGLPArchiver.get(), pGBArchiver, pModel.get());
		}
	}
}
