#pragma once

#include "MPSDef.h"
#include "thrust/device_vector.h"
#include "../MCUDA_Lib/MCUDAGLBuffer.h"

namespace mps
{
	class Tree
	{};
	template<class T>
	class VirtualTree : public Tree
	{
	public:
		VirtualTree()
		{}

	public:
		T& GetParam() { return m_param; }
		const T& GetParam() const { return m_param; }

	private:
		T m_param;
	};
};