#pragma once
#include "../MPS_Object/MPSDef.h"
#include "../MCUDA_Lib/MCUDAHelper.cuh"

#define CONST_W						1.5666814710608447114749495456982 // 315.0 / (64.0 * M_PI)
#define CONST_G						-14.323944878270580219199538703526 // -45.0 / M_PI
#define CONST_CUBICW				2.5464790894703253723021402139602 // 8.0 / M_PI
#define CONST_CUBICG				15.278874536821952233812841283761 // 48.0 / M_PI
#define CONST_LAPLACIAN				14.323944878270580219199538703526 // 45.0 / M_PI
#define CONST_COHESION				10.185916357881301489208560855841 // 32.0 / (M_PI);
#define CONST_ADHESION				0.007

namespace mps
{
	namespace kernel::sph
	{
		MCUDA_HOST_DEVICE_FUNC constexpr REAL WKernel(REAL ratio)
		{
			if (ratio < 0.0 || ratio >= 1.0)
				return 0.0;

		#if 1
			REAL tmp = 1.0 - ratio * ratio;
			return CONST_W * tmp * tmp * tmp;
		#else
			REAL result;
			if (ratio <= 0.5) {
				REAL tmp2 = ratio * ratio;
				result = 6.0 * tmp2 * (ratio - 1.0) + 1.0;
			}
			else {
				result = 1.0 - ratio;
				result = 2.0 * result * result * result;
			}
			return CONST_CUBICW * result;
		#endif
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL GKernel(REAL ratio)
		{
			if (ratio < 1.0e-40 || ratio >= 1.0)
				return 0.0;
		#if 1
			REAL tmp = 1.0 - ratio;
			return CONST_G * tmp * tmp;
		#else
			REAL result;
			if (ratio <= 0.5)
				result = ratio * (3.0 * ratio - 2.0);
			else {
				result = 1.0 - ratio;
				result = -result * result;
			}
			return CONST_CUBICG * result;
		#endif
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL LaplacianKernel(REAL ratio)
		{
			if (ratio < 0.0 || ratio >= 1.0)
				return 0.0;

			REAL tmp = 1.0 - ratio;
			return CONST_LAPLACIAN * tmp;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL cohesionKernel(REAL ratio)
		{
			if (ratio <= 1.0e-40 || ratio >= 1.0)
				return 0.0;

			REAL result = (1.0 - ratio) * ratio;
			result = result * result * result;
			if (ratio <= 0.5)
				result += result - 0.015625;

			return CONST_COHESION * result;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL addhesionKernel(REAL ratio)
		{
			if (ratio <= 0.5 || ratio >= 1.0)
				return 0.0;

			REAL result = pow(-4.0 * ratio * ratio + 6.0 * ratio - 2.0, 0.25);
			return CONST_ADHESION * result;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL selfCohesionKernel(REAL ratio)
		{
			if (ratio <= 0.5 || ratio >= 1.0)
				return 0.0;

			REAL result = (1.0 - ratio * 0.5) * ratio * 0.5;
			result = result * result * result;

			return CONST_COHESION * result;
		}

		MCUDA_HOST_DEVICE_FUNC constexpr REAL WKernel(REAL dist, REAL invh)
		{
			return WKernel(dist * invh) * invh * invh * invh;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL GKernel(REAL dist, REAL invh)
		{
			REAL tmp = invh * invh;
			return GKernel(dist * invh) * tmp * tmp;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL LaplacianKernel(REAL dist, REAL invh)
		{
			REAL tmp = invh * invh;

			//return LaplacianKernel(dist * invh) * tmp * tmp * invh;

			dist = dist * invh;
			return -GKernel(dist) / (dist * dist + 0.01) * tmp * tmp * invh;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL cohesionKernel(REAL dist, REAL invh)
		{
			return cohesionKernel(dist * invh) * invh * invh * invh;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL addhesionKernel(REAL dist, REAL invh)
		{
			return addhesionKernel(dist * invh) * invh * invh * invh;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL selfCohesionKernel(REAL dist, REAL invh)
		{
			return selfCohesionKernel(dist * invh) * invh * invh * invh;
		}

		MCUDA_HOST_DEVICE_FUNC constexpr REAL WKernel(REAL dist, REAL invhi, REAL invhj)
		{
			return (WKernel(dist, invhi) + WKernel(dist, invhj)) * 0.5;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL GKernel(REAL dist, REAL invhi, REAL invhj)
		{
			return (GKernel(dist, invhi) + GKernel(dist, invhj)) * 0.5;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL LaplacianKernel(REAL dist, REAL invhi, REAL invhj)
		{
			return (LaplacianKernel(dist, invhi) + LaplacianKernel(dist, invhj)) * 0.5;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL cohesionKernel(REAL dist, REAL invhi, REAL invhj)
		{
			return (cohesionKernel(dist, invhi) + cohesionKernel(dist, invhj)) * 0.5;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL addhesionKernel(REAL dist, REAL invhi, REAL invhj)
		{
			return (addhesionKernel(dist, invhi) + addhesionKernel(dist, invhj)) * 0.5;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL selfCohesionKernel(REAL dist, REAL invhi, REAL invhj)
		{
			return (selfCohesionKernel(dist, invhi) + selfCohesionKernel(dist, invhj)) * 0.5;
		}
		MCUDA_HOST_DEVICE_FUNC constexpr REAL GKernel(REAL dist, REAL hi, REAL hj, REAL invhi, REAL invhj)
		{
			return (hi * GKernel(dist, invhi) + hj * GKernel(dist, invhj)) * 0.5;
		}
	}
}