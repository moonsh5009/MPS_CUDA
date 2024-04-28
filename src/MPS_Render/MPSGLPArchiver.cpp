#include "stdafx.h"
#include "MPSGLPArchiver.h"

#include "../MCUDA_Lib/MGLShader.h"

void mps::rndr::GLPArchiver::Initalize()
{
	for (int i = 0; i < static_cast<int>(def::ProgramDef::Size); i++)
	{
		const auto& [vertPath, fragPath] = def::aShaderInfo[i];
		mgl::Shader vShader, fShader;
		vShader.Create(vertPath, GL_VERTEX_SHADER);
		fShader.Create(fragPath, GL_FRAGMENT_SHADER);

		m_aProgram[i].Create();
		m_aProgram[i].CompileShader(vShader);
		m_aProgram[i].CompileShader(fShader);
	}
}