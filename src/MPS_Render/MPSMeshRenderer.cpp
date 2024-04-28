#include "stdafx.h"
#include "MPSMeshRenderer.h"

#include "../MCUDA_Lib/MGLVertexArray.h"

#include "../MPS_Object/MPSGBArchiver.h"
#include "../MPS_Object/MPSObstacleModel.h"

#include "MPSGLPArchiver.h"

void mps::rndr::MeshRenderer::Initalize(const GLPArchiver* pGLPArchiver, const GBArchiver* pGBArchiver)
{
	const auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Mesh);
	const auto iVertShader = program.GetVertexShader();

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, pGBArchiver->m_cameraBuffer.GetID());
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, pGBArchiver->m_lightBuffer.GetID());
}

void mps::rndr::MeshRenderer::Draw(const GLPArchiver* pGLPArchiver, const GBArchiver* pGBArchiver, const Model* pModel) const
{
	auto pObstacleModel = static_cast<const mps::ObstacleModel*>(pModel);
	{
		/*auto pMeshObject = pObstacleModel->GetTarget<mps::MeshObject>();

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pMeshObject->m_face.GetID());
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pMeshObject->m_pos.GetID());
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pMeshObject->m_color.GetID());
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, pMeshObject->m_backColor.GetID());

		auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Mesh);
		program.Bind();

		glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(pMeshObject->GetSize()) * 3);

		program.Unbind();*/
	}
	{
		/*auto pParticleObject = static_cast<mps::ParticleObject*>(pObstacleModel->GetSubObject());

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pParticleObject->m_pos.GetID());
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pParticleObject->m_radius.GetID());
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pParticleObject->m_color.GetID());

		auto& program = pGLPArchiver->GetProgram(mps::rndr::def::ProgramDef::Particle);
		program.Bind();

		glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(pParticleObject->GetSize()));

		program.Unbind();*/
	}
}