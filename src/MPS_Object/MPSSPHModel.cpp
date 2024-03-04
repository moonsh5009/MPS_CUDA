#include "stdafx.h"
#include "MPSSPHModel.h"

mps::SPHModel::SPHModel(std::unique_ptr<SPHMaterial>&& pMaterial, std::unique_ptr<SPHObject>&& pObject, std::unique_ptr<SpatialHash>&& pTree) :
	mps::Model{ std::move(pMaterial), std::move(pObject), std::move(pTree) }
{

}
