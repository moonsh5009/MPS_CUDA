#pragma once

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	class Model;
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