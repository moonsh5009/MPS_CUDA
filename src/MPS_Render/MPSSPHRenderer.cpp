#include "stdafx.h"
#include "MPSSPHRenderer.h"

#include "../MCUDA_Lib/MGLVertexArray.h"

#include "../MPS_Object/MPSGBArchiver.h"
#include "../MPS_Object/MPSSPHModel.h"

#include "MPSGLPArchiver.h"

void mps::rndr::SPHRenderer::Initalize(const mps::rndr::GLPArchiver* pGLPArchiver, const mps::GBArchiver* pGBArchiver)
{
	const auto&[vShader, fShader] = pGLPArchiver->GetShader(mps::rndr::def::ProgramDef::Particle);

	const auto glCameraIndex = glGetUniformBlockIndex(vShader.GetID(), "uCamera");
	const auto glLightIndex = glGetUniformBlockIndex(vShader.GetID(), "uLight");

	glUniformBlockBinding(vShader.GetID(), glCameraIndex, 1);
	glUniformBlockBinding(vShader.GetID(), glLightIndex, 2);

	glBindBufferBase(GL_UNIFORM_BUFFER, 1, pGBArchiver->m_cameraBuffer.GetID());
	glBindBufferBase(GL_UNIFORM_BUFFER, 2, pGBArchiver->m_lightBuffer.GetID());
}

void mps::rndr::SPHRenderer::Draw(const mps::rndr::GLPArchiver* pGLPArchiver, const mps::GBArchiver* pGBArchiver, const mps::Model* pModel) const
{
	auto pSPHObject = static_cast<mps::SPHObject*>(pModel->GetTarget());

	auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Particle);
	program.Bind();

	mgl::VertexArray vao;
	vao.Create();
	vao.AddVertexBuffer(pSPHObject->m_pos);

	vao.Bind();
	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pSPHObject->GetSize()));
	vao.Unbind();

	program.Unbind();
}
