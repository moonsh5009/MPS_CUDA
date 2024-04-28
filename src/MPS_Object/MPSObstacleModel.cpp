#include "stdafx.h"
#include "MPSObstacleModel.h"

mps::ObstacleModel::ObstacleModel(
	std::unique_ptr<MeshObject>&& pObject,
	std::unique_ptr<BoundaryParticleObject>&& pBoundaryParticle,
	std::unique_ptr<MeshMaterial>&& pMaterial,
	std::unique_ptr<SpatialHash>&& pSpatilHash) :
	SuperClass{ std::move(pObject) }
{
	SetSubObject(0u, std::move(pBoundaryParticle));
	SetMaterial(0u, std::move(pMaterial));
	SetTree(0u, std::move(pSpatilHash));
}