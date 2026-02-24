#pragma once

#include "IDeviceMemory.h"

#include "HeaderPre.h"

namespace mcuda
{
	class __MY_EXT_CLASS__ DeviceMemory : public IDeviceMemory
	{
    public:
        DeviceMemory() = default;
        ~DeviceMemory() override;
        DeviceMemory(const DeviceMemory&) = default;
        DeviceMemory(DeviceMemory&& src) noexcept;
        DeviceMemory& operator=(const DeviceMemory&) = delete;
        DeviceMemory& operator=(DeviceMemory&& src) noexcept;

        void Create(size_t byteSize) override;
        void Destroy() override;
	};
}

#include "HeaderPost.h"