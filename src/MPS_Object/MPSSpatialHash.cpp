#include "stdafx.h"
#include "MPSSpatialHash.h"

namespace
{
	constexpr auto uObjectSizeByteLength = sizeof(size_t);
	constexpr auto uCeilSizeByteLength = sizeof(REAL);
	constexpr auto uHashSizeByteLength = sizeof(glm::uvec3);

	constexpr auto uObjectSizeOffset = 0ull;
	constexpr auto uCeilSizeOffset = uObjectSizeOffset + uObjectSizeByteLength;
	constexpr auto uHashSizeOffset = uCeilSizeOffset + uCeilSizeByteLength;
}

mps::SpatialHash::SpatialHash() : mps::VirtualTree<SpatialHashParam>{}
{
	SetParam({ 0ull, 0.0f, glm::uvec3{ 0u, 0u, 0u } });
}

void mps::SpatialHash::SetObjectSize(const size_t size)
{
	GetHost().objSize = size;
	GetDevice().CopyFromHost(GetHost(), uObjectSizeOffset, uObjectSizeOffset, uObjectSizeByteLength);

	m_keys.Resize(size);
	m_IDs.Resize(size);
}

void mps::SpatialHash::SetCeilSize(const REAL size)
{
	GetHost().ceilSize = size;
	GetDevice().CopyFromHost(GetHost(), uCeilSizeOffset, uCeilSizeOffset, uCeilSizeByteLength);
}

void mps::SpatialHash::SetHashSize(const glm::uvec3& size)
{
	GetHost().hashSize = size;
	GetDevice().CopyFromHost(GetHost(), uHashSizeOffset, uHashSizeOffset, uHashSizeByteLength);

	const auto hashCeilSize = size.x * size.y * size.z;
	m_iStarts.Resize(hashCeilSize);
	m_iEnds.Resize(hashCeilSize);
}

void mps::SpatialHash::InitHashID()
{
}

void mps::SpatialHash::InitHash(const mcuda::gl::Buffer<REAL3>& dArPos)
{
}

void mps::SpatialHash::ColorTest(const mcuda::gl::Buffer<REAL3>& dArPos, const mcuda::gl::Buffer<REAL4>& dArColor)
{
}