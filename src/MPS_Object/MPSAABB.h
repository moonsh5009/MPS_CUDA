#pragma once

#include "MPSDef.h"
#include "../MCUDA_Lib/MCUDAHelper.h"

namespace mps
{
	class AABB
	{
	public:
		MCUDA_HOST_DEVICE_FUNC AABB() : min{ DBL_MAX }, max{ -DBL_MAX } {}
		MCUDA_HOST_DEVICE_FUNC ~AABB() {}

	public:
		MCUDA_HOST_DEVICE_FUNC void Reset()
		{
			min = REAL3{ DBL_MAX };
			max = REAL3{ -DBL_MAX };
		}
		MCUDA_HOST_DEVICE_FUNC void AddPoint(const REAL3& v)
		{
			if (min.x > v.x) min.x = v.x;
			else if (max.x < v.x) max.x = v.x;
			if (min.y > v.y) min.y = v.y;
			else if (max.y < v.y) max.y = v.y;
			if (min.z > v.z) min.z = v.z;
			else if (max.z < v.z) max.z = v.z;
		}

	public:
		REAL3 min;
		REAL3 max;
	};
}