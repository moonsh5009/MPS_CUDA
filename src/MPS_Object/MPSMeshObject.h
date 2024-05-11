#pragma once

#include "MPSMeshParam.h"
#include "MPSObject.h"
#include "MPSAABB.h"

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
		virtual void ResizeVertex(const size_t size);
		constexpr size_t GetVertexSize() { return m_position.GetSize(); }

		void LoadMesh(const std::string_view& filePath, const REAL3& vCenter, const REAL3& vSize, REAL density);

	protected:
		virtual std::shared_ptr<ObjectResource> GenerateDeviceResource();

	public:
		mcuda::gl::Buffer<glm::fvec4> m_backColor;
		mcuda::gl::Buffer<glm::uvec3> m_face;

		mcuda::gl::Buffer<REAL3> m_faceNorm;
		thrust::device_vector<REAL> m_faceArea;
		
		thrust::device_vector<uint32_t> m_nbFaces;
		thrust::device_vector<uint32_t> m_nbFacesIdx;

		thrust::device_vector<uint32_t> m_nbNodes;
		thrust::device_vector<uint32_t> m_nbNodesIdx;

		thrust::device_vector<uint32_t> m_rTri;

		thrust::device_vector<uint32_t> m_shortestEdgeID;
		thrust::device_vector<uint32_t> m_samplingParticleSize;

	private:
		void Translate(const AABB& aabb, const REAL3& vCenter, const REAL3& vSize);
		void InitFaceInfo(REAL density);
		void buildAdjacency();
	};
}

#include "HeaderPost.h"