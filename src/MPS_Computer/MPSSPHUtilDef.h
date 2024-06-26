#pragma once

#include "../MPS_Object/MPSDef.h"

#define SPH_TIMER_PRINT		1
#define SPH_DEBUG_PRINT		0

namespace mps::kernel::SPH
{
	constexpr auto nFullBlockSize = 1024u;
	constexpr auto nBlockSize = 128u;
}

namespace mps
{
	struct PhysicsParam;
	struct SPHParam;
	struct BoundaryParticleParam;
	struct NeiParam;
	struct SPHMaterialParam;
	struct MeshMaterialParam;
}