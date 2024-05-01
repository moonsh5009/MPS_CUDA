#pragma once

#include "MPSTree.h"
#include "MPSSpatialHashParam.h"

#include "HeaderPre.h"

namespace mps
{
	class Object;
	class ObjectParam;
	class SPHParam;
	class BoundaryParticleParam;
	struct NeiBuffer
	{
		thrust::device_vector<uint32_t> data;
		thrust::device_vector<uint32_t> idx;
	};
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

		std::optional<std::tuple<const uint32_t*, const uint32_t*>> GetNeighborhood(const ObjectParam& objParam) const;

	public:
		void UpdateHash(const ObjectParam& objParam);
		void ZSort(ObjectParam& objParam);
		void ZSort(SPHParam& sphParam);
		void ZSort(BoundaryParticleParam& boundaryParticleParam);

		void BuildNeighorhood(const ObjectParam& objParam, REAL radius);
		void BuildNeighorhood(const ObjectParam& objParam, REAL radius, const ObjectParam& refObjParam, const SpatialHash* pRefHash);

	private:
		thrust::device_vector<uint32_t> m_key;
		thrust::device_vector<uint32_t> m_ID;
		thrust::device_vector<uint32_t> m_startIdx;
		thrust::device_vector<uint32_t> m_endIdx;

		std::unordered_map<size_t, NeiBuffer> m_mapNei;
	};
};

#include "HeaderPost.h"