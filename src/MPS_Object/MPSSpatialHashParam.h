#pragma once

#include "../MCUDA_Lib/MCUDAHelper.h"
#include "MPSDef.h"

namespace mps
{
	class SpatialHashParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC SpatialHashParam() : m_objSize{ 0ull }, m_ceilSize{ 0.0 }, m_hashSize{ 0u },
			m_pKey{ nullptr }, m_pID{ nullptr }, m_pStartIdx{ nullptr }, m_pEndIdx{ nullptr }
		{}
		MCUDA_HOST_DEVICE_FUNC ~SpatialHashParam() {}

	public:
		MCUDA_HOST_DEVICE_FUNC size_t GetObjectSize() const { return m_objSize; }
		MCUDA_HOST_DEVICE_FUNC REAL GetCeilSize() const { return m_ceilSize; }
		MCUDA_HOST_DEVICE_FUNC const glm::uvec3& GetHashSize() const { return m_hashSize; }

		MCUDA_DEVICE_FUNC uint32_t& Key(uint32_t idx) { return m_pKey[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& ID(uint32_t idx) { return m_pID[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& StartIdx(uint32_t idx) { return m_pStartIdx[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& EndIdx(uint32_t idx) { return m_pEndIdx[idx]; }

		MCUDA_DEVICE_FUNC uint32_t Key(uint32_t idx) const { return m_pKey[idx]; }
		MCUDA_DEVICE_FUNC uint32_t ID(uint32_t idx) const { return m_pID[idx]; }
		MCUDA_DEVICE_FUNC uint32_t StartIdx(uint32_t idx) const { return m_pStartIdx[idx]; }
		MCUDA_DEVICE_FUNC uint32_t EndIdx(uint32_t idx) const { return m_pEndIdx[idx]; }

	public:
		MCUDA_HOST_FUNC void SetObjectSize(uint32_t objSize) { m_objSize = objSize; }
		MCUDA_HOST_FUNC void SetCeilSize(REAL ceilSize) { m_ceilSize = ceilSize; }
		MCUDA_HOST_FUNC void SetHashSize(const glm::uvec3& hashSize) { m_hashSize = hashSize; }

		MCUDA_HOST_FUNC void SetKeyArray(uint32_t* pKey) { m_pKey = pKey; }
		MCUDA_HOST_FUNC void SetIDArray(uint32_t* pID) { m_pID = pID; }
		MCUDA_HOST_FUNC void SetStartIdxArray(uint32_t* pStartIdx) { m_pStartIdx = pStartIdx; }
		MCUDA_HOST_FUNC void SetEndIdxArray(uint32_t* pEndIdx) { m_pEndIdx = pEndIdx; }

	public:
		MCUDA_DEVICE_FUNC glm::ivec3 GetGridPos(const REAL3& x) const;
		MCUDA_DEVICE_FUNC uint32_t GetGridIndex(const glm::ivec3& p) const;
		MCUDA_DEVICE_FUNC uint32_t GetGridZIndex(const glm::ivec3& p) const;

		template<class Fn>
		MCUDA_DEVICE_FUNC void Research(const REAL3& pos, Fn func) const;

	private:
		size_t m_objSize;
		REAL m_ceilSize;
		glm::uvec3 m_hashSize;

		uint32_t* MCUDA_RESTRICT m_pKey;
		uint32_t* MCUDA_RESTRICT m_pID;
		uint32_t* MCUDA_RESTRICT m_pStartIdx;
		uint32_t* MCUDA_RESTRICT m_pEndIdx;
	};
}