#include "stdafx.h"
#include "MPSObject.h"

mps::Object::Object() : m_size{ 0ull }
{}

void mps::Object::Clear()
{
	m_size = 0ull;

	m_color.Clear();
	m_pos.Clear();
	m_mass.clear();
	m_velocity.clear();
	m_force.clear();
}

void mps::Object::Resize(const size_t size)
{
	m_size = size;

	m_color.Resize(size);
	m_pos.Resize(size);
	m_mass.resize(size);
	m_velocity.resize(size);
	m_force.resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::Object::GenerateDeviceResource()
{
	auto posRes = m_pos.GetDeviceResource();
	if (!posRes) return {};

	auto colorRes = m_color.GetDeviceResource();
	if (!colorRes) return {};

	return std::make_shared<ObjectResource>(m_size,
		std::move(colorRes.value()),
		std::move(posRes.value()),
		thrust::raw_pointer_cast(m_mass.data()),
		thrust::raw_pointer_cast(m_velocity.data()),
		thrust::raw_pointer_cast(m_force.data()));
}