#include "stdafx.h"
#include "LightUniform.h"

#include "../MCore_interface/IScene.h"

using namespace mcore::render;

IMPLEMENT_RENDER_UNIFORM(LightUniform, RenderUniformType::LIGHT, vk::ShaderStageFlagBits::eFragment)

LightUniform::LightUniform(IScene* pScene)
    : RenderUniformBase{ pScene }
	, m_hostData{}
{}

void LightUniform::Initialize()
{
    m_hostData.pos = { glm::normalize(glm::vec3{ 1.f, 1.f, 1.f }), 1.f };
    m_hostData.color.ambient = { 1.f, 1.f, 1.f, 1.f };
    m_hostData.color.diffuse = { 1.f, 1.f, 1.f, 1.f };
    m_hostData.color.specular = { 1.f, 1.f, 1.f, 1.f };

    SetDirty();
}

bool LightUniform::Update(const vk::CommandBuffer& commandBuffer)
{
    if (!IsDirty())
        return false;

    GetUBO().CopyFromHostAndUpload(commandBuffer, m_hostData);

    ClearDirty();
    return true;
}