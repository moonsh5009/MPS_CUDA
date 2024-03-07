#pragma once

#include "MPSRenderer.h"
#include "../MCUDA_Lib/MCUDAGLBuffer.h"

#include "HeaderPre.h"

namespace mps
{
	namespace rndr
	{
		class __MY_EXT_CLASS__ SPHRenderer : public Renderer
		{
		public:
			virtual void Initalize(const GLPArchiver*, const GBArchiver*) override;
			virtual void Draw(const GLPArchiver*, const GBArchiver*, const Model*) const override;

		private:
			mgl::Buffer<glm::fvec2> m_vbo;
		};
	}
};

#include "HeaderPost.h"