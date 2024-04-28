#pragma once

#include "MPSModel.h"

#include "MPSMeshObject.h"
#include "MPSBoundaryParticleObject.h"
#include "MPSMeshMaterial.h"
#include "MPSSpatialHash.h"

#include "HeaderPre.h"

namespace mps
{
	enum class ObstacleSubObjectIdx : uint32_t
	{
		BoundaryParticle,
		Size,
	};
	enum class ObstacleTreeIdx : uint32_t
	{
		SpatialHash,
		Size
	};
	class __MY_EXT_CLASS__ ObstacleModel : public VirtualModel<static_cast<uint32_t>(ObstacleSubObjectIdx::Size), 1, static_cast<uint32_t>(ObstacleTreeIdx::Size)>
	{
		using SuperClass = VirtualModel<static_cast<uint32_t>(ObstacleSubObjectIdx::Size), 1, static_cast<uint32_t>(ObstacleTreeIdx::Size)>;
	public:
		ObstacleModel(std::unique_ptr<MeshObject>&& pObject,
			std::unique_ptr<BoundaryParticleObject>&& pBoundaryParticle,
			std::unique_ptr<MeshMaterial>&& pMaterial,
			std::unique_ptr<SpatialHash>&& pSpatialHash);
	};
};

#include "HeaderPost.h"