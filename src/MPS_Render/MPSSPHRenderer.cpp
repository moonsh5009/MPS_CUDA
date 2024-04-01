#include "stdafx.h"
#include "MPSSPHRenderer.h"

#include "../MCUDA_Lib/MGLVertexArray.h"

#include "../MPS_Object/MPSGBArchiver.h"
#include "../MPS_Object/MPSSPHModel.h"

#include "MPSGLPArchiver.h"

#if 0
void mps::rndr::SPHRenderer::Initalize(const mps::rndr::GLPArchiver* pGLPArchiver, const mps::GBArchiver* pGBArchiver)
{
	const auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Particle);
	const auto iVertShader = program.GetVertexShader();

	const auto glCameraIndex = glGetUniformBlockIndex(iVertShader, "uCamera");
	const auto glLightIndex = glGetUniformBlockIndex(iVertShader, "uLight");

	glUniformBlockBinding(iVertShader, glCameraIndex, 1);
	glUniformBlockBinding(iVertShader, glLightIndex, 2);

	glBindBufferBase(GL_UNIFORM_BUFFER, 1, pGBArchiver->m_cameraBuffer.GetID());
	glBindBufferBase(GL_UNIFORM_BUFFER, 2, pGBArchiver->m_lightBuffer.GetID());

	m_vbo.Resize(
		{
			glm::fvec2{ -1.0, -1.0 },
			glm::fvec2{ 1.0, -1.0 },
			glm::fvec2{ -1.0, 1.0 },
			glm::fvec2{ 1.0, 1.0 },
		});
}

void mps::rndr::SPHRenderer::Draw(const mps::rndr::GLPArchiver* pGLPArchiver, const mps::GBArchiver* pGBArchiver, const mps::Model* pModel) const
{
	auto pSPHObject = static_cast<mps::SPHObject*>(pModel->GetTarget());

	auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Particle);
	program.Bind();

	mgl::VertexArray vao;
	vao.Create();
	vao.AddVertexBuffer(m_vbo, mgl::VertexStepMode::Vertex);
	vao.AddVertexBuffer(pSPHObject->m_pos, mgl::VertexStepMode::Instance);
	vao.AddVertexBuffer(pSPHObject->m_radius, mgl::VertexStepMode::Instance);
	//vao.AddVertexBuffer(pSPHObject->m_color, mgl::VertexStepMode::Instance);

	vao.Bind();
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(pSPHObject->GetSize()));
	vao.Unbind();

	program.Unbind();
}

#else
void mps::rndr::SPHRenderer::Initalize(const mps::rndr::GLPArchiver* pGLPArchiver, const mps::GBArchiver* pGBArchiver)
{
	const auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Particle);
	const auto iVertShader = program.GetVertexShader();

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pGBArchiver->m_cameraBuffer.GetID());
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, pGBArchiver->m_lightBuffer.GetID());

	m_vbo.Resize(
		{
			glm::fvec2{ -1.0, -1.0 },
			glm::fvec2{ 1.0, -1.0 },
			glm::fvec2{ -1.0, 1.0 },
			glm::fvec2{ 1.0, 1.0 },
		});
}

void mps::rndr::SPHRenderer::Draw(const mps::rndr::GLPArchiver* pGLPArchiver, const mps::GBArchiver* pGBArchiver, const mps::Model* pModel) const
{
	auto pSPHObject = static_cast<mps::SPHObject*>(pModel->GetTarget());

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pSPHObject->m_pos.GetID());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pSPHObject->m_radius.GetID());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pSPHObject->m_color.GetID());

	auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Particle);
	program.Bind();

	mgl::VertexArray vao;
	vao.Create();
	vao.AddVertexBuffer(m_vbo, mgl::VertexStepMode::Vertex);

	vao.Bind();
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(pSPHObject->GetSize()));
	vao.Unbind();

	program.Unbind();
}

#endif