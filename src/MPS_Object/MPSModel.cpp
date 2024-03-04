#include "stdafx.h"
#include "MPSModel.h"

mps::Model::Model(std::unique_ptr<Material>&& pMaterial, std::unique_ptr<Object>&& pObject, std::unique_ptr<Tree>&& pTree) :
	m_pMaterial{ std::move(pMaterial) },
	m_pTree{ std::move(pTree) },
	m_pObject{ std::move(pObject) }
{
}