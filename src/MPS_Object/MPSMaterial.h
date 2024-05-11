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
		VirtualMaterial()
		{
		}

	public:
		constexpr T& GetParam() { return m_param; }
		constexpr const T& GetParam() const { return m_param; }

	private:
		T m_param;
	};
};