#pragma once

#include "MPSMeshParam.h"
#include "MPSObject.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ MeshObject : public Object
	{
	public:
		MeshObject();
		~MeshObject() = default;
		MeshObject(const MeshObject&) = delete;
		MeshObject(MeshObject&&) = default;
		MeshObject& operator=(const MeshObject&) = delete;
		MeshObject& operator=(MeshObject&&) = default;

	public:
		virtual void Clear() override;
		virtual void Resize(const size_t size) override;
		virtual void ResizeFace(const size_t size);

	protected:
		virtual std::shared_ptr<ObjectResource> GenerateDeviceResource();

	public:
		mcuda::gl::Buffer<glm::uvec3> m_face;
		
		thrust::device_vector<uint32_t> m_nbFaces;
		thrust::device_vector<uint32_t> m_idxNbFaces;

		thrust::device_vector<uint32_t> m_nbNodes;
		thrust::device_vector<uint32_t> m_idxNbNodes;
	};
}

#include "HeaderPost.h"