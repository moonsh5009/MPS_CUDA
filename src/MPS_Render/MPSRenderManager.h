#pragma once

#include "MPSGLPArchiver.h"

#include "HeaderPre.h"

namespace mps
{
	class Model;
	class GBArchiver;
	namespace rndr
	{
		class Renderer;
		class __MY_EXT_CLASS__ RenderManager
		{
		public:
			RenderManager();
			~RenderManager() = default;
			RenderManager(const RenderManager&) = delete;
			RenderManager(RenderManager&&) = delete;
			RenderManager& operator=(const RenderManager&) = delete;
			RenderManager& operator=(RenderManager&&) = delete;

		public:
			void Initalize(const mps::GBArchiver*);
			void AddModel(const std::shared_ptr<Model>&);
			void Draw(const mps::GBArchiver*);

		private:
			std::unique_ptr<GLPArchiver> m_pGLPArchiver;

		private:
			enum RENDERER
			{
				Mesh,
				SPH,
				Size
			};

			std::vector<std::shared_ptr<Renderer>> m_aRenderer;
			std::unordered_map<size_t, RENDERER> m_mRndrID;
			std::vector<std::unordered_set<std::shared_ptr<Model>>> m_aModel;
		};
	}
}

#include "HeaderPost.h"