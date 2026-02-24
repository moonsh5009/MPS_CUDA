#include "stdafx.h"
#include "RenderConfigUniform.h"

#include "../MCore_interface/IScene.h"

using namespace mcore::render;

IMPLEMENT_RENDER_UNIFORM(RenderConfigUniform, RenderUniformType::RENDER_CONFIG, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)

RenderConfigUniform::RenderConfigUniform(IScene* pScene)
    : RenderUniformBase{ pScene }
	, m_hostData{}
{}

void RenderConfigUniform::Initialize()
{
    m_hostData.const_color = { 1.f, 1.f, 1.f, 1.f };
    m_hostData.color_type = ColorType::Attribute;
    m_hostData.transparent_type = ColorType::Attribute;
	m_hostData.model_type = 0;
	m_hostData.use_point_size = 0;
	m_hostData.point_size = 1.f;
	m_hostData.use_line_width = 0;
	m_hostData.line_width = 1.f;

    SetDirty();
}

bool RenderConfigUniform::Update(const vk::CommandBuffer& commandBuffer)
{
    if (!IsDirty())
        return false;

    GetUBO().CopyFromHostAndUpload(commandBuffer, m_hostData);

    ClearDirty();
    return true;
}

void RenderConfigUniform::SetConstColor(const glm::vec4& color)
{
    m_hostData.const_color = color;
	SetDirty();
}

void RenderConfigUniform::SetColorType(ColorType type)
{
	m_hostData.color_type = type;
	SetDirty();
}

void RenderConfigUniform::SetTransparentType(ColorType type)
{
    m_hostData.transparent_type = type;
	SetDirty();
}

void RenderConfigUniform::SetModelType(RenderModelType type)
{
    m_hostData.model_type = to_uint(type);
	SetDirty();
}

void RenderConfigUniform::SetUsePointSize(bool use)
{
    m_hostData.use_point_size = use ? 1 : 0;
	SetDirty();
}

void RenderConfigUniform::SetPointSize(float size)
{
    m_hostData.point_size = size;
	SetDirty();
}

void RenderConfigUniform::SetUseLineWidth(bool use)
{
    m_hostData.use_line_width = use ? 1 : 0;
	SetDirty();
}

void RenderConfigUniform::SetLineWidth(float width)
{
    m_hostData.line_width = width;
	SetDirty();
}