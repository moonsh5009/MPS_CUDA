#include "stdafx.h"
#include "MGLVertexArray.h"

mgl::VertexArray::VertexArray() : m_id{ 0 }, m_index{ 0 }
{
}

mgl::VertexArray::~VertexArray()
{
	Delete();
}

void mgl::VertexArray::Create()
{
	Delete();

	glCreateVertexArrays(1, &m_id);
}

void mgl::VertexArray::Delete()
{
	if (!IsAssigned()) return;

	glDeleteVertexArrays(1, &m_id);
	m_id = 0;
	m_index = 0;
}

void mgl::VertexArray::Bind() const
{
	if (!IsAssigned()) return;

	glBindVertexArray(m_id);
}

void mgl::VertexArray::Unbind() const
{
	glBindVertexArray(0);
}
