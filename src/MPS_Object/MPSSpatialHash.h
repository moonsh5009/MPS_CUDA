#pragma once

#include "MPSTree.h"
#include "MPSSpatialHashParam.h"

#include "HeaderPre.h"

namespace mps
{
	struct ParticleParam;
	struct SPHParam;
	struct BoundaryParticleParam;

	struct NeiParam
	{
		const uint32_t* pID;
		const uint32_t* pIdx;
	};

	class __MY_EXT_CLASS__ SpatialHash : public VirtualTree<SpatialHashParam>
	{
	public:
		SpatialHash();

	public:
		void SetObjectSize(size_t size);
		void SetCeilSize(REAL size);
		void SetHashSize(const glm::uvec3& size);

		constexpr uint32_t GetObjectSize() const { return GetParam().objSize; };
		constexpr REAL GetCeilSize() const { return GetParam().ceilSize; };
		constexpr const glm::uvec3& GetHashSize() const { return GetParam().hashSize; };

		constexpr const thrust::device_vector<uint32_t>& GetHashKeys() const { return m_key; };
		constexpr const thrust::device_vector<uint32_t>& GetHashIDs() const { return m_ID; };
		constexpr const thrust::device_vector<uint32_t>& GetStartIdx() const { return m_startIdx; };
		constexpr const thrust::device_vector<uint32_t>& GetEndIdx() const { return m_endIdx; };

		std::optional<NeiParam> GetNeighborhood(const ParticleParam& objParam) const;

	public:
		void UpdateHash(const ParticleParam& particleParam);
		void ZSort(ParticleParam& particleParam);
		void ZSort(SPHParam& sphParam);
		void ZSort(BoundaryParticleParam& boundaryParticleParam);

		void BuildNeighorhood(const ParticleParam& particleParam);
		void BuildNeighorhood(const ParticleParam& particleParam, const ParticleParam& refParticleParam, const SpatialHash* pRefHash);

	private:
		thrust::device_vector<uint32_t> m_key;
		thrust::device_vector<uint32_t> m_ID;
		thrust::device_vector<uint32_t> m_startIdx;
		thrust::device_vector<uint32_t> m_endIdx;

		std::unordered_map<size_t, std::tuple<thrust::device_vector<uint32_t>, thrust::device_vector<uint32_t>>> m_mapNei;
	};
};

#include "HeaderPost.h"