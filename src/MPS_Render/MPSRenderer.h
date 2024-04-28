#pragma once

#include "../MPS_Object/MPSSPHModel.h"

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	namespace rndr
	{
		class GLPArchiver;
		class __MY_EXT_CLASS__ Renderer
		{
		public:
			virtual void Initalize(const GLPArchiver*, const GBArchiver*) = 0;
			virtual void Draw(const GLPArchiver*, const GBArchiver*, const Model*) const = 0;
		};
	}
};

#include "HeaderPost.h"