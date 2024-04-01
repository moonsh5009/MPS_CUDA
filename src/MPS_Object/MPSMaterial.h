#pragma once

#include "MPSDef.h"
#include "../MCUDA_Lib/MCUDAGLBuffer.h"

namespace mps
{
	class Material
	{
	};
	template<class T>
	class VirtualMaterial : public Material
	{
	public:
		VirtualMaterial() : m_deviceParam{ 1 }
		{
		}

	public:
		constexpr T& GetHost() { return m_hostParam; }
		constexpr mcuda::gl::Buffer<T>& GetDevice() { return m_deviceParam; }
		constexpr const T& GetHost() const { return m_hostParam; }
		constexpr const mcuda::gl::Buffer<T>& GetDevice() const { return m_deviceParam; }

		void SetParam(const T& param)
		{
			m_hostParam = param;
			m_deviceParam.CopyFromHost(m_hostParam);
		}

	private:
		T m_hostParam;
		mcuda::gl::Buffer<T> m_deviceParam;
	};
};