#pragma once

#include "MPSTree.h"

#include "HeaderPre.h"

namespace mps
{
	struct SpatialHashParam
	{
		size_t objSize;
		REAL ceilSize;
		glm::uvec3 hashSize;
	};
	class __MY_EXT_CLASS__ SpatialHash : public VirtualTree<SpatialHashParam>
	{
	public:
		SpatialHash();

	public:
		void SetObjectSize(const size_t size);
		void SetCeilSize(const REAL size);
		void SetHashSize(const glm::uvec3& size);

		constexpr uint32_t GetObjectSize() const { return GetHost().objSize; };
		constexpr REAL GetCeilSize() const { return GetHost().ceilSize; };
		constexpr glm::uvec3 GetHashSize() const { return GetHost().hashSize; };

		constexpr const mcuda::gl::Buffer<uint32_t>& GetHashKeys() const { return m_keys; };
		constexpr const mcuda::gl::Buffer<uint32_t>& GetHashIDs() const { return m_IDs; };
		constexpr const mcuda::gl::Buffer<uint32_t>& GetStartIdx() const { return m_iStarts; };
		constexpr const mcuda::gl::Buffer<uint32_t>& GetEndIdx() const { return m_iEnds; };

	public:
		void InitHashID();
		void InitHash(const mcuda::gl::Buffer<REAL3>& dArPos);
		void ColorTest(const mcuda::gl::Buffer<REAL3>& dArPos, const mcuda::gl::Buffer<REAL4>& dArColor);

	private:
		mcuda::gl::Buffer<uint32_t> m_keys;
		mcuda::gl::Buffer<uint32_t> m_IDs;
		mcuda::gl::Buffer<uint32_t> m_iStarts;
		mcuda::gl::Buffer<uint32_t> m_iEnds;
	};
};

#include "HeaderPost.h"