#pragma once

#include "../MPS_Object/MPSDef.h"
#include "../MCUDA_Lib/MCUDAHelper.cuh"

namespace mps::device
{
	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL3 AtomicAdd(REAL3* ptr, const REAL3& val)
	{
		REAL* onePtr = reinterpret_cast<REAL*>(ptr);
		return
		{
			mcuda::util::AtomicAdd(onePtr, val.x),
			mcuda::util::AtomicAdd(onePtr + 1, val.y),
			mcuda::util::AtomicAdd(onePtr + 2, val.z)
		};
	}
}