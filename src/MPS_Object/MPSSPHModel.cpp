#include "stdafx.h"
#include "MPSSPHModel.h"

mps::SPHModel::SPHModel(std::unique_ptr<SPHObject>&& pObject,
	std::unique_ptr<SPHMaterial>&& pMaterial,
	std::unique_ptr<SpatialHash>&& pTree) :
	SuperClass{ std::move(pObject) }
{
	SetMaterial(0u, std::move(pMaterial));
	SetTree(0u, std::move(pTree));
}