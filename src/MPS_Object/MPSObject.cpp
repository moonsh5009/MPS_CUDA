#include "stdafx.h"
#include "MPSObject.h"

mps::Object::Object() : m_size{ 0ull }
{
}

void mps::Object::Clear()
{
	m_size = 0ull;
}

void mps::Object::Resize(const size_t size)
{
	m_size = size;
}

std::shared_ptr<mps::ObjectResource> mps::Object::GetObjectResource()
{
	return std::make_shared<ObjectResource>(m_size);
}
