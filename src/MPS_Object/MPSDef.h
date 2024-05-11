#pragma once

#include "glm/glm.hpp"

#if 1
typedef double				REAL;
typedef glm::dvec2			REAL2;
typedef glm::dvec3			REAL3;
typedef glm::dvec4			REAL4;
typedef glm::dmat2			REAL2x2;
typedef glm::dmat3			REAL3x3;
typedef glm::dmat4			REAL4x4;
typedef unsigned long long	REAL_INT;
#else
typedef float				REAL;
typedef glm::fvec2			REAL2;
typedef glm::fvec3			REAL3;
typedef glm::fvec4			REAL4;
typedef glm::fmat2			REAL2x2;
typedef glm::fmat3			REAL3x3;
typedef glm::fmat4			REAL4x4;
typedef unsigned int		REAL_INT;
#endif

namespace mps::device::SPH
{
	constexpr auto H_RATIO = 4.0;
}