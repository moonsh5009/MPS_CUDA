#pragma once

#include "MPSTree.h"
#include "MPSSpatialHashParam.h"

#include "HeaderPre.h"

namespace mps
{
	class ObjectParam;
	class SPHParam;
	class BoundaryParticleParam;
	class __MY_EXT_CLASS__ SpatialHash : public VirtualTree<SpatialHashParam>
	{
	public:
		SpatialHash();

	public:
		void SetObjectSize(const size_t size);
		void SetCeilSize(const REAL size);
		void SetHashSize(const glm::uvec3& size);

		uint32_t GetObjectSize() const { return GetParam().GetObjectSize(); };
		REAL GetCeilSize() const { return GetParam().GetCeilSize(); };
		const glm::uvec3& GetHashSize() const { return GetParam().GetHashSize(); };

		const thrust::device_vector<uint32_t>& GetHashKeys() const { return m_key; };
		const thrust::device_vector<uint32_t>& GetHashIDs() const { return m_ID; };
		const thrust::device_vector<uint32_t>& GetStartIdx() const { return m_startIdx; };
		const thrust::device_vector<uint32_t>& GetEndIdx() const { return m_endIdx; };

	public:
		void UpdateHash(const mps::ObjectParam& objParam);
		void ZSort(mps::ObjectParam& objParam);
		void ZSort(mps::SPHParam& sphParam);
		void ZSort(mps::BoundaryParticleParam& boundaryParticleParam);

	private:
		thrust::device_vector<uint32_t> m_key;
		thrust::device_vector<uint32_t> m_ID;
		thrust::device_vector<uint32_t> m_startIdx;
		thrust::device_vector<uint32_t> m_endIdx;
	};
};

#include "HeaderPost.h"