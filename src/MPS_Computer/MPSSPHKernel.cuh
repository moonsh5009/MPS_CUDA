#pragma once

#include "MPSBaseUtil.cuh"

#define USE_CUBIC_KERNEL		1

namespace mps::device::SPH
{
	constexpr auto CONST_W = static_cast<REAL>(1.5666814710608447114749495456982); // 315.0 / (64.0 * PI)
	constexpr auto CONST_G = static_cast<REAL>(-14.323944878270580219199538703526); // -45.0 / PI
	constexpr auto CONST_CUBICW = static_cast<REAL>(2.5464790894703253723021402139602); // 8.0 / PI
	constexpr auto CONST_CUBICG = static_cast<REAL>(5.0929581789406507446042804279205); // 16.0 / PI
	constexpr auto CONST_STW = static_cast<REAL>(7.2756545413437867780061148970292); // 8.0 / PI * 10 / 7 * 2
	constexpr auto CONST_STG = static_cast<REAL>(21.826963624031360334018344691088); // 16.0 / PI * 10 / 7 * 3
	constexpr auto SPH_EPSILON = static_cast<REAL>(1.0e-20);

	template<class Fn>
	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC void NeighborSearch(uint32_t id, const uint32_t* pNei, const uint32_t* pNeiIdx, Fn func)
	{
		const auto iEnd = pNeiIdx[id + 1u];
	#pragma unroll
		for (auto idx = pNeiIdx[id]; idx < iEnd; ++idx)
		{
			func(pNei[idx]);
		}
	}

	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL WKernel(REAL dist, REAL invh)
	{
		const auto ratio = dist * invh;
		if (ratio >= static_cast<REAL>(1.0))
			return static_cast<REAL>(0.0);

	#if USE_CUBIC_KERNEL
		const auto mult = CONST_CUBICW * invh * invh * invh;
		if (ratio >= static_cast<REAL>(0.5))
		{
			const auto temp = static_cast<REAL>(1.0) - ratio;
			return static_cast<REAL>(2.0) * temp * temp * temp * mult;
		}
		const auto ratio2 = ratio * ratio;
		return ((static_cast<REAL>(6.0) * ratio - static_cast<REAL>(6.0)) * ratio2 + static_cast<REAL>(1.0)) * mult;
	#else
		const auto tmp = static_cast<REAL>(1.0) - ratio * ratio;
		return CONST_W * invh * invh * invh * tmp * tmp * tmp;
	#endif
	}
	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL GKernel(REAL dist, REAL invh)
	{
		const auto ratio = dist * invh;
		if (ratio < SPH_EPSILON || ratio >= static_cast<REAL>(1.0))
			return static_cast<REAL>(0.0);

	#if USE_CUBIC_KERNEL
		const auto invh2 = invh * invh;
		const auto mult = CONST_CUBICG * invh2 * invh2;
		if (ratio > static_cast<REAL>(0.5))
		{
			const auto temp = static_cast<REAL>(1.0) - ratio;
			return -static_cast<REAL>(3.0) * temp * temp * mult;
		}
		return (static_cast<REAL>(9.0) * ratio * ratio - static_cast<REAL>(6.0) * ratio) * mult;
	#else
		const auto invh2 = invh * invh;
		const auto tmp = static_cast<REAL>(1.0) - ratio;
		return CONST_G * invh2 * invh2 * tmp * tmp;
	#endif
	}

	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL STWKernel(REAL dist, REAL invh)
	{
		const auto ratio = mcuda::util::max(dist * invh, static_cast<REAL>(0.5));
		if (ratio >= static_cast<REAL>(1.0))
			return static_cast<REAL>(0.0);

		const auto temp = (static_cast<REAL>(1.0) - ratio) * invh;
		return temp * temp * temp * CONST_STW;
	}
	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL STGKernel(REAL dist, REAL invh)
	{
		const auto ratio = dist * invh;
		if (ratio <= 0.5 || ratio >= static_cast<REAL>(1.0))
			return static_cast<REAL>(0.0);

		const auto invh2 = invh * invh;
		const auto temp = (static_cast<REAL>(1.0) - ratio) * invh2;
		return -temp * temp * CONST_STG;
	}

	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL WKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (WKernel(dist, invhi) + WKernel(dist, invhj)) * static_cast<REAL>(0.5);
	}
	MCUDA_INLINE_FUNC MCUDA_DEVICE_FUNC REAL GKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (GKernel(dist, invhi) + GKernel(dist, invhj)) * static_cast<REAL>(0.5);
	}
}