#include "stdafx.h"
#include "DeviceMemory.h"

#include "MCudaUtil.cuh"

using namespace mcuda;

mcuda::DeviceMemory::~DeviceMemory()
{
	DeviceMemory::Destroy();
}

DeviceMemory::DeviceMemory(DeviceMemory&& src) noexcept
{
    *this = std::move(src);
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& src) noexcept
{
    if (this != &src)
    {
        DeviceMemory::Destroy();

        IDeviceMemory::operator=(std::move(src));
    }
    return *this;
}

void DeviceMemory::Create(size_t byteSize)
{
	Destroy();

	m_byteSize = byteSize;
	if (byteSize == 0)
		return;

	CUDA_CHECK(cudaMalloc((void**)&m_rawPtr, byteSize));
}

void DeviceMemory::Destroy()
{
	if (m_rawPtr)
	{
		CUDA_CHECK(cudaFree(m_rawPtr));
		m_rawPtr = nullptr;
	}
	m_byteSize = 0;
}