#pragma once

#include "../MCUDA_Lib/MCUDAHelper.h"
#include "MPSDef.h"

namespace mps
{
	struct SpatialHashParam
	{
		size_t objSize;
		REAL ceilSize;
		glm::uvec3 hashSize;

		uint32_t* pKey;
		uint32_t* pID;
		uint32_t* pStartIdx;
		uint32_t* pEndIdx;
	};
}