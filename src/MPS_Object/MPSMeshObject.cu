#include "stdafx.h"
#include "MPSMeshObject.cuh"

#include <thrust/sort.h>
#include <fstream>

mps::MeshObject::MeshObject() : Object{}
{}

void mps::MeshObject::Clear()
{
	Object::Clear();

	m_backColor.Clear();
	m_face.Clear();
	m_faceNorm.Clear();
	m_faceArea.clear();
	m_nbFaces.clear();
	m_nbFacesIdx.clear();
	m_nbNodes.clear();
	m_nbNodesIdx.clear();
	m_rTri.clear();
}

void mps::MeshObject::Resize(const size_t size)
{
	m_size = size;

	m_color.Resize(size);
	m_backColor.Resize(size);

	m_face.Resize(size);
	m_faceNorm.Resize(size);
	m_faceArea.resize(size);

	m_rTri.resize(size);
	m_shortestEdgeID.resize(size);
	m_samplingParticleSize.resize(size, 0u);
}

void mps::MeshObject::ResizeVertex(const size_t size)
{
	m_pos.Resize(size);
	m_mass.resize(size);
	m_velocity.resize(size);
	m_force.resize(size);

	m_nbNodesIdx.resize(size + 1, 0u);
	m_nbFacesIdx.resize(size + 1, 0u);
}

void mps::MeshObject::LoadMesh(const std::string_view& filePath, const REAL3& vCenter, const REAL3& vSize, REAL density)
{
	std::vector<glm::uvec3> faces;
	std::vector<REAL3> vertices;
	AABB aabb;

	std::ifstream fin;
	fin.open(filePath.data());
	if (fin.is_open())
	{
		while (!fin.eof())
		{
			std::string head;
			fin >> head;
			if (head.length() > 1) continue;
			if (head[0] == 'v')
			{
				REAL3 x;
				fin >> x.x >> x.y >> x.z;
				vertices.emplace_back(x);
				aabb.AddPoint(x);
			}
			else if (head[0] == 'f')
			{
				glm::uvec3 x;
				fin >> x.x >> x.y >> x.z;
				faces.emplace_back(x - 1u);
			}
		}
		fin.close();
	}
	if (faces.empty() || vertices.empty())
	{
		printf("Error : Mesh_init : Object Load Error\n");
		exit(1);
		return;
	}
	Resize(faces.size());
	ResizeVertex(vertices.size());
	m_face.CopyFromHost(faces);
	m_pos.CopyFromHost(vertices);

	Translate(aabb, vCenter, vSize);
	InitFaceInfo(density);
	buildAdjacency();
}

std::shared_ptr<mps::ObjectResource> mps::MeshObject::GenerateDeviceResource()
{
	auto pSuperParam = Object::GenerateDeviceResource();
	if (!pSuperParam) return {};

	auto optFaceRes = m_face.GetDeviceResource();
	if (!optFaceRes) return {};

	auto optFaceNormRes = m_faceNorm.GetDeviceResource();
	if (!optFaceNormRes) return {};

	return std::make_shared<MeshResource>(pSuperParam,
		std::move(optFaceRes.value()),
		std::move(optFaceNormRes.value()),
		thrust::raw_pointer_cast(m_faceArea.data()),
		thrust::raw_pointer_cast(m_nbFaces.data()),
		thrust::raw_pointer_cast(m_nbFacesIdx.data()),
		thrust::raw_pointer_cast(m_nbNodes.data()),
		thrust::raw_pointer_cast(m_nbNodesIdx.data()),
		thrust::raw_pointer_cast(m_rTri.data()),
		thrust::raw_pointer_cast(m_shortestEdgeID.data()),
		thrust::raw_pointer_cast(m_samplingParticleSize.data()));
}

void mps::MeshObject::Translate(const AABB& aabb, const REAL3& vCenter, const REAL3& vSize)
{
	constexpr auto nBlockSize = 1024u;

	const auto optPosRes = m_pos.GetDeviceResource();
	if (!optPosRes) return;

	const auto oriCenter = (aabb.min + aabb.max) * 0.5;
	const auto oriSize = aabb.max - aabb.min;
	const auto scale = REAL3
	{
		oriSize.x > 1.0e-10 ? vSize.x / oriSize.x : 0.0,
		oriSize.y > 1.0e-10 ? vSize.y / oriSize.y : 0.0,
		oriSize.z > 1.0e-10 ? vSize.z / oriSize.z : 0.0
	};

	TranslateMesh_kernel << < mcuda::util::DivUp(GetVertexSize(), nBlockSize), nBlockSize >> >
		(optPosRes->GetData(), GetVertexSize(), oriCenter, vCenter, scale);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::MeshObject::InitFaceInfo(REAL density)
{
	constexpr auto nBlockSize = 256u;

	const auto optFaceRes = m_face.GetDeviceResource();
	if (!optFaceRes) return;

	const auto optPosRes = m_pos.GetDeviceResource();
	if (!optPosRes) return;

	const auto optFaceNormRes = m_faceNorm.GetDeviceResource();
	if (!optFaceNormRes) return;

	ComputeMeshFaceInfo_kernel << < mcuda::util::DivUp(GetSize(), nBlockSize), nBlockSize >> >
		(optFaceRes->GetData(), GetSize(), optPosRes->GetData(), optFaceNormRes->GetData(), thrust::raw_pointer_cast(m_faceArea.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	const auto area = thrust::reduce(m_faceArea.begin(), m_faceArea.end(), static_cast<REAL>(0.0), thrust::plus<REAL>());
	const auto mass = area * density;
	
	thrust::fill(m_mass.begin(), m_mass.end(), mass / static_cast<REAL>(GetVertexSize()));
}

struct SortUint2CMP
{
	MCUDA_HOST_DEVICE_FUNC constexpr bool operator()(const uint2& a, const uint2& b) const
	{
		if (a.x != b.x)
			return a.x < b.x;
		return a.y < b.y;
	}
};

struct TransformUint2CMP
{
	MCUDA_HOST_DEVICE_FUNC constexpr uint32_t operator()(const uint2& a) const
	{
		return a.y;
	}
};

void mps::MeshObject::buildAdjacency()
{
	constexpr auto nBlockSize = 256u;

	const auto optFaceRes = m_face.GetDeviceResource();
	if (!optFaceRes) return;

	thrust::device_vector<uint32_t> d_nbFacesID{ GetVertexSize() + 1u, 0u };
	thrust::device_vector<uint32_t> d_nbNodesID{ GetVertexSize() + 1u, 0u };
	thrust::device_vector<uint2> d_vertexID;

	ComputeNbFacesSize_kernel << < mcuda::util::DivUp(GetSize(), nBlockSize), nBlockSize >> >
		(optFaceRes->GetData(), GetSize(), thrust::raw_pointer_cast(d_nbFacesID.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::exclusive_scan(d_nbFacesID.begin(), d_nbFacesID.end(), d_nbFacesID.begin());
	thrust::copy(d_nbFacesID.begin(), d_nbFacesID.end(), m_nbFacesIdx.begin());
	m_nbFaces.resize(d_nbFacesID.back());
	d_vertexID.resize(d_nbFacesID.back());

	ComputeNbFaces_kernel << < mcuda::util::DivUp(GetSize(), nBlockSize), nBlockSize >> >
		(optFaceRes->GetData(), GetSize(), thrust::raw_pointer_cast(d_vertexID.data()), thrust::raw_pointer_cast(d_nbFacesID.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort(d_vertexID.begin(), d_vertexID.end(), SortUint2CMP());
	thrust::transform(d_vertexID.begin(), d_vertexID.end(), m_nbFaces.begin(), TransformUint2CMP());

	RTriangleBuild_kernel << < mcuda::util::DivUp(GetSize(), nBlockSize), nBlockSize >> >
		(optFaceRes->GetData(), GetSize(), thrust::raw_pointer_cast(m_nbFaces.data()), thrust::raw_pointer_cast(m_nbFacesIdx.data()), thrust::raw_pointer_cast(m_rTri.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	ComputeNbNodesSize_kernel << < mcuda::util::DivUp(GetSize(), nBlockSize), nBlockSize >> >
		(optFaceRes->GetData(), GetSize(), thrust::raw_pointer_cast(m_rTri.data()), thrust::raw_pointer_cast(d_nbNodesID.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::exclusive_scan(d_nbNodesID.begin(), d_nbNodesID.end(), d_nbNodesID.begin());
	thrust::copy(d_nbNodesID.begin(), d_nbNodesID.end(), m_nbNodesIdx.begin());
	m_nbNodes.resize(d_nbNodesID.back());
	d_vertexID.resize(d_nbNodesID.back());

	ComputeNbNodes_kernel << < mcuda::util::DivUp(GetSize(), nBlockSize), nBlockSize >> >
		(optFaceRes->GetData(), GetSize(), thrust::raw_pointer_cast(m_rTri.data()), thrust::raw_pointer_cast(d_vertexID.data()), thrust::raw_pointer_cast(d_nbNodesID.data()));
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort(d_vertexID.begin(), d_vertexID.end(), SortUint2CMP());
	thrust::transform(d_vertexID.begin(), d_vertexID.end(), m_nbNodes.begin(), TransformUint2CMP());
}
