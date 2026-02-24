#include "stdafx.h"
#include "IDeviceMemory.h"

#include "MCudaUtil.cuh"

using namespace mcuda;

IDeviceMemory::IDeviceMemory(IDeviceMemory&& src) noexcept
{
	*this = std::move(src);
}

IDeviceMemory& IDeviceMemory::operator=(IDeviceMemory&& src) noexcept
{
	if (this != &src)
	{
		m_byteSize = std::exchange(src.m_byteSize, 0);
		m_rawPtr = std::exchange(src.m_rawPtr, nullptr);
	}
	return *this;
}

void IDeviceMemory::_CopyFromDevice(const void* src, size_t size, size_t offset, size_t srcOffset) const
{
	CUDA_CHECK(cudaMemcpy(
		static_cast<char*>(m_rawPtr) + offset,
		static_cast<const char*>(src) + srcOffset,
		size, cudaMemcpyDeviceToDevice));
}

void IDeviceMemory::_CopyFromDevice(const IDeviceMemory& src, size_t size, size_t offset, size_t srcOffset) const
{
	_CopyFromDevice(src.GetRawPointer(), size, offset, srcOffset);
}

void IDeviceMemory::_CopyToDevice(void* dst, size_t size, size_t offset, size_t dstOffset) const
{
	CUDA_CHECK(cudaMemcpy(
		static_cast<char*>(dst) + dstOffset,
		static_cast<const char*>(m_rawPtr) + offset,
		size, cudaMemcpyDeviceToDevice));
}

void IDeviceMemory::_CopyToDevice(const IDeviceMemory& dst, size_t size, size_t offset, size_t dstOffset) const
{
	_CopyToDevice(dst.GetRawPointer(), size, offset, dstOffset);
}

void IDeviceMemory::_CopyFromHost(const void* src, size_t size, size_t offset, size_t srcOffset) const
{
	CUDA_CHECK(cudaMemcpy(
		static_cast<char*>(m_rawPtr) + offset,
		static_cast<const char*>(src) + srcOffset,
		size, cudaMemcpyHostToDevice));
}

void IDeviceMemory::_CopyToHost(void* dst, size_t size, size_t offset, size_t dstOffset) const
{
	CUDA_CHECK(cudaMemcpy(
		static_cast<char*>(dst) + dstOffset,
		static_cast<const char*>(m_rawPtr) + offset,
		size, cudaMemcpyDeviceToHost));
}

#ifdef MCORE_USE_CUDA
void IDeviceMemory::_CopyFromDeviceAsync(const void* src, size_t size, size_t offset, size_t srcOffset, cudaStream_t stream) const
{
	CUDA_CHECK(cudaMemcpyAsync(
		static_cast<char*>(m_rawPtr) + offset,
		static_cast<const char*>(src) + srcOffset,
		size, cudaMemcpyDeviceToDevice, stream));
}

void IDeviceMemory::_CopyFromDeviceAsync(const IDeviceMemory& src, size_t size, size_t offset, size_t srcOffset, cudaStream_t stream) const
{
	_CopyFromDeviceAsync(src.GetRawPointer(), size, offset, srcOffset, stream);
}

void IDeviceMemory::_CopyToDeviceAsync(void* dst, size_t size, size_t offset, size_t dstOffset, cudaStream_t stream) const
{
	CUDA_CHECK(cudaMemcpyAsync(
		static_cast<char*>(dst) + dstOffset,
		static_cast<const char*>(m_rawPtr) + offset,
		size, cudaMemcpyDeviceToDevice, stream));
}

void IDeviceMemory::_CopyToDeviceAsync(const IDeviceMemory& dst, size_t size, size_t offset, size_t dstOffset, cudaStream_t stream) const
{
	_CopyToDeviceAsync(dst.GetRawPointer(), size, offset, dstOffset, stream);
}

void IDeviceMemory::_CopyFromHostAsync(const void* src, size_t size, size_t offset, size_t srcOffset, cudaStream_t stream) const
{
	CUDA_CHECK(cudaMemcpyAsync(
		static_cast<char*>(m_rawPtr) + offset,
		static_cast<const char*>(src) + srcOffset,
		size, cudaMemcpyHostToDevice, stream));
}

void IDeviceMemory::_CopyToHostAsync(void* dst, size_t size, size_t offset, size_t dstOffset, cudaStream_t stream) const
{
	CUDA_CHECK(cudaMemcpyAsync(
		static_cast<char*>(dst) + dstOffset,
		static_cast<const char*>(m_rawPtr) + offset,
		size, cudaMemcpyDeviceToHost, stream));
}
#endif