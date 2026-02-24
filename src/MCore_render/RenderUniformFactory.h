#pragma once

#include "../MCore_interface/IRenderUniform.h"

#include <array>
#include <memory>
#include <functional>

namespace mcore
{
	class IScene;
	namespace render
	{
		class RenderUniformFactory
		{
			using Func = std::function<std::unique_ptr<IRenderUniform>(IScene*)>;
			using BuildFunc = std::function<mvk::BindGroupLayoutBuilder&&(mvk::BindGroupLayoutBuilder&&)>;

		public:
			static RenderUniformFactory& Instance();

			bool Registry(RenderUniformType type, Func&& func, BuildFunc&& buildFunc);
			RenderUniformArray Build(IScene* pScene) const;
			mvk::BindGroupLayout BuildBindGroupLayout() const;

		private:
			std::array<Func, static_cast<size_t>(RenderUniformType::Size)> m_funcs;
			std::array<BuildFunc, static_cast<size_t>(RenderUniformType::Size)> m_buildFuncs;
		};
	}
}