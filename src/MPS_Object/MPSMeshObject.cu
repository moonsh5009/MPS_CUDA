#include "stdafx.h"
#include "MPSMeshObject.h"

mps::MeshObject::MeshObject() : Object{}
{}

void mps::MeshObject::Clear()
{
	Object::Clear();

	m_face.Clear();
	m_nbFaces.clear();
	m_idxNbFaces.clear();
	m_nbNodes.clear();
	m_idxNbNodes.clear();
}

void mps::MeshObject::Resize(const size_t size)
{
	Object::Resize(size);

	m_idxNbNodes.resize(size);
	m_idxNbFaces.resize(size);
}

void mps::MeshObject::ResizeFace(const size_t size)
{
	m_face.Resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::MeshObject::GenerateDeviceResource()
{
	auto pSuperParam = Object::GenerateDeviceResource();
	if (!pSuperParam) return {};

	auto optFaceRes = m_face.GetDeviceResource();
	if (!optFaceRes) return {};

	return std::make_shared<MeshResource>(pSuperParam,
		std::move(optFaceRes.value()),
		thrust::raw_pointer_cast(m_nbFaces.data()),
		thrust::raw_pointer_cast(m_idxNbFaces.data()),
		thrust::raw_pointer_cast(m_nbNodes.data()),
		thrust::raw_pointer_cast(m_idxNbNodes.data()));
}
