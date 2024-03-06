#pragma once

#include "../MCUDA_Lib/MCUDAGLBuffer.h"
#include "MPSUniformDef.h"

#include "HeaderPre.h"

namespace mgpu
{
	class MGPUCore;
}
namespace mps
{
	class __MY_EXT_CLASS__ GBArchiver
	{
	public:
		GBArchiver() = default;
		~GBArchiver() = default;
		GBArchiver(const GBArchiver&) = delete;
		GBArchiver(GBArchiver&&) = default;
		GBArchiver& operator=(const GBArchiver&) = delete;
		GBArchiver& operator=(GBArchiver&&) = default;

	public:
		void Initalize();

		void UpdateCamera(const mps::CameraParam& camera);
		void UpdateLight(const mps::LightParam& light);
		void UpdateLightPos(const glm::vec3& lightPos);

	public:
		mcuda::gl::Buffer<mps::CameraParam> m_cameraBuffer;
		mcuda::gl::Buffer<mps::LightParam> m_lightBuffer;
	};
}

#include "HeaderPost.h"