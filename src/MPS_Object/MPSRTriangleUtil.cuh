#pragma once
#include "../MCUDA_Lib/MCUDAHelper.cuh"

namespace mps
{
	MCUDA_DEVICE_FUNC void SetRTriVertex(uint32_t& info, uint32_t n)
	{
		info |= (1u << n);
	}
	MCUDA_DEVICE_FUNC void SetRTriEdge(uint32_t& info, uint32_t n)
	{
		info |= (1u << (n + 3u));
	}
	MCUDA_DEVICE_FUNC bool RTriVertex(uint32_t info, uint32_t n)
	{
		return (info >> n) & 1u;
	}
	MCUDA_DEVICE_FUNC bool RTriEdge(uint32_t info, uint32_t n)
	{
		return (info >> (n + 3u)) & 1u;
	}
}