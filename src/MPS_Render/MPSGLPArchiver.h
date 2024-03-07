#pragma once

#include "../MCUDA_Lib/MGLProgram.h"
#include "../MCUDA_Lib/MGLShader.h"

#include "HeaderPre.h"

namespace mps
{
	namespace rndr
	{
		namespace def
		{
			enum class ProgramDef : int
			{
				Particle,
				Size,
			};

			using ShaderInfo = std::tuple<std::string_view, std::string_view>;
			constexpr std::array<ShaderInfo, static_cast<int>(ProgramDef::Size)> aShaderInfo
			{
				ShaderInfo{ "particleShader.vert", "particleShader.frag" },
			};
		}

		class __MY_EXT_CLASS__ GLPArchiver
		{
		public:
			void Initalize();

		public:
			mgl::Program& GetProgram(const def::ProgramDef program) { return m_aProgram[static_cast<int>(program)]; }
			const mgl::Program& GetProgram(const def::ProgramDef program) const { return m_aProgram[static_cast<int>(program)]; }
			
		private:
			std::array<mgl::Program, static_cast<int>(def::ProgramDef::Size)> m_aProgram;
		};
	}
}

#include "HeaderPost.h"
