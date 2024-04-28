#pragma once

#include "MPSRenderer.h"

#include "HeaderPre.h"

namespace mps
{
	namespace rndr
	{
		class __MY_EXT_CLASS__ MeshRenderer : public Renderer
		{
		public:
			virtual void Initalize(const GLPArchiver* pGLPArchiver, const GBArchiver* pGBArchiver) override;
			virtual void Draw(const GLPArchiver* pGLPArchiver, const GBArchiver* pGBArchiver, const Model* pModel) const override;
		};
	}
};

#include "HeaderPost.h"