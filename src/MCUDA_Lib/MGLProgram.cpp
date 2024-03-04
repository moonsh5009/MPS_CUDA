#include "stdafx.h"
#include "MGLProgram.h"

#include "MGLShader.h"

mgl::Program::Program() : m_id{ 0u }
{
}

mgl::Program::~Program()
{
    Destroy();
}

void mgl::Program::Create()
{
    Destroy();

    m_id = glCreateProgram();
}

bool mgl::Program::CompileShader(const mgl::Shader& pShader)
{
    auto iShader = pShader.GetID();

    glAttachShader(m_id, iShader);
    glLinkProgram(m_id);

    GLint isSuccess = 0;
    glGetProgramiv(m_id, GL_LINK_STATUS, &isSuccess);
    if (isSuccess == 0)
    {
        char temp[256];
        glGetProgramInfoLog(m_id, 256, 0, temp);
        //TRACE("Failed to link program:\n%s\n", temp);
        OutputDebugString(temp);
        assert(false);
        Destroy();
    }

    return isSuccess != 0;
}

void mgl::Program::Destroy()
{
    if (!IsAssigned()) return;

    glDeleteProgram(m_id);
}

void mgl::Program::Bind() const
{
    if (!IsAssigned()) return;

    glUseProgram(m_id);
}

void mgl::Program::Unbind() const
{
    glUseProgram(0);
}