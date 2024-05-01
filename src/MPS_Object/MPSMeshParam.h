#pragma once

#include "MPSObjectParam.h"

namespace mps
{
	class MeshParam : public ObjectParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC MeshParam() : ObjectParam{},
			m_pFace{ nullptr }, m_pFaceNorm{ nullptr }, m_pFaceArea{ nullptr }, m_pNbFaces{ nullptr }, m_pNbFacesIdx{ nullptr }, m_pNbNodes{ nullptr }, m_pNbNodesIdx{ nullptr },
			m_pRTri{ nullptr }, m_pShortestEdgeID{ nullptr }, m_pSamplingParticleSize{ nullptr } {}
		MCUDA_HOST_DEVICE_FUNC ~MeshParam() {}

	public:
		MCUDA_DEVICE_FUNC glm::uvec3& Face(uint32_t idx) { return m_pFace[idx]; }
		MCUDA_DEVICE_FUNC REAL3& FaceNorm(uint32_t idx) { return m_pFaceNorm[idx]; }
		MCUDA_DEVICE_FUNC REAL& FaceArea(uint32_t idx) { return m_pFaceArea[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& NbFaces(uint32_t idx) { return m_pNbFaces[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& NbFacesIdx(uint32_t idx) { return m_pNbFacesIdx[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& NbNodes(uint32_t idx) { return m_pNbNodes[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& NbNodesIdx(uint32_t idx) { return m_pNbNodesIdx[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& RTriangle(uint32_t idx) { return m_pRTri[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& ShortEdgeId(uint32_t idx) { return m_pShortestEdgeID[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& SamplingParticleSize(uint32_t idx) { return m_pSamplingParticleSize[idx]; }

		MCUDA_DEVICE_FUNC glm::uvec3 Face(uint32_t idx) const { return m_pFace[idx]; }
		MCUDA_DEVICE_FUNC REAL3 FaceNorm(uint32_t idx) const { return m_pFaceNorm[idx]; }
		MCUDA_DEVICE_FUNC REAL FaceArea(uint32_t idx) const { return m_pFaceArea[idx]; }
		MCUDA_DEVICE_FUNC uint32_t NbFaces(uint32_t idx) const { return m_pNbFaces[idx]; }
		MCUDA_DEVICE_FUNC uint32_t NbFacesIdx(uint32_t idx) const { return m_pNbFacesIdx[idx]; }
		MCUDA_DEVICE_FUNC uint32_t NbNodes(uint32_t idx) const { return m_pNbNodes[idx]; }
		MCUDA_DEVICE_FUNC uint32_t NbNodesIdx(uint32_t idx) const { return m_pNbNodesIdx[idx]; }
		MCUDA_DEVICE_FUNC uint32_t RTriangle(uint32_t idx) const { return m_pRTri[idx]; }
		MCUDA_DEVICE_FUNC uint32_t ShortEdgeId(uint32_t idx) const { return m_pShortestEdgeID[idx]; }
		MCUDA_DEVICE_FUNC uint32_t SamplingParticleSize(uint32_t idx) const { return m_pSamplingParticleSize[idx]; }

	public:
		MCUDA_HOST_FUNC glm::uvec3* GetFaceArray() const { return m_pFace; }
		MCUDA_HOST_FUNC REAL3* GetFaceNormArray() const { return m_pFaceNorm; }
		MCUDA_HOST_FUNC REAL* GetFaceAreaArray() const { return m_pFaceArea; }
		MCUDA_HOST_FUNC uint32_t* GetNbFacesArray() const { return m_pNbFaces; }
		MCUDA_HOST_FUNC uint32_t* GetNbFacesIdxArray() const { return m_pNbFacesIdx; }
		MCUDA_HOST_FUNC uint32_t* GetNbNodesArray() const { return m_pNbNodes; }
		MCUDA_HOST_FUNC uint32_t* GetNbNodesIdxArray() const { return m_pNbNodesIdx; }
		MCUDA_HOST_FUNC uint32_t* GetRTriangleArray() const { return m_pRTri; }
		MCUDA_HOST_FUNC uint32_t* GetShortestEdgeIdArray() const { return m_pShortestEdgeID; }
		MCUDA_HOST_FUNC uint32_t* GetSamplingParticleSizeArray() const { return m_pSamplingParticleSize; }

		MCUDA_HOST_FUNC void SetFaceArray(glm::uvec3* pFace) { m_pFace = pFace; }
		MCUDA_HOST_FUNC void SetFaceNormArray(REAL3* pFaceNorm) { m_pFaceNorm = pFaceNorm; }
		MCUDA_HOST_FUNC void SetFaceAreaArray(REAL* pFaceArea) { m_pFaceArea = pFaceArea; }
		MCUDA_HOST_FUNC void SetNbFacesArray(uint32_t* pNbFaces) { m_pNbFaces = pNbFaces; }
		MCUDA_HOST_FUNC void SetNbFacesIdxArray(uint32_t* pNbFacesIdx) { m_pNbFacesIdx = pNbFacesIdx; }
		MCUDA_HOST_FUNC void SetNbNodesArray(uint32_t* pNbNodes) { m_pNbNodes = pNbNodes; }
		MCUDA_HOST_FUNC void SetNbNodesIdxArray(uint32_t* pNbNodesIdx) { m_pNbNodesIdx = pNbNodesIdx; }
		MCUDA_HOST_FUNC void SetRTriangleArray(uint32_t* pRTri) { m_pRTri = pRTri; }
		MCUDA_HOST_FUNC void SetShortestEdgeIdArray(uint32_t* pShortestEdgeID) { m_pShortestEdgeID = pShortestEdgeID; }
		MCUDA_HOST_FUNC void SetSamplingParticleSizeArray(uint32_t* pSamplingParticleSize) { m_pSamplingParticleSize = pSamplingParticleSize; }

	private:
		glm::uvec3* MCUDA_RESTRICT m_pFace;
		REAL3* MCUDA_RESTRICT m_pFaceNorm;
		REAL* MCUDA_RESTRICT m_pFaceArea;
		uint32_t* MCUDA_RESTRICT m_pNbFaces;
		uint32_t* MCUDA_RESTRICT m_pNbFacesIdx;
		uint32_t* MCUDA_RESTRICT m_pNbNodes;
		uint32_t* MCUDA_RESTRICT m_pNbNodesIdx;
		uint32_t* MCUDA_RESTRICT m_pRTri;
		uint32_t* MCUDA_RESTRICT m_pShortestEdgeID;
		uint32_t* MCUDA_RESTRICT m_pSamplingParticleSize;
	};

	struct MeshResource : public ObjectResource
	{
	public:
		MeshResource() = delete;
		MeshResource(std::shared_ptr<ObjectResource> pSuper, mcuda::gl::DeviceResource<glm::uvec3>&& face, mcuda::gl::DeviceResource<REAL3>&& faceNorm,
			REAL* faceArea, uint32_t* nbFaces, uint32_t* nbFacesIdx, uint32_t* nbNodes, uint32_t* nbNodesIdx,
			uint32_t* rTri, uint32_t* shortestEdgeId, uint32_t* samplingParticleSize) :
			ObjectResource{ std::move(*pSuper) }, m_face{ std::move(face) }, m_faceNorm{ std::move(faceNorm) },
			m_faceArea{ faceArea }, m_nbFaces{ nbFaces }, m_nbFacesIdx{ nbFacesIdx }, m_nbNodes{ nbNodes }, m_nbNodesIdx{ nbNodesIdx },
			m_rTri{ rTri }, m_shortestEdgeId{ shortestEdgeId }, m_samplingParticleSize{ samplingParticleSize }
		{}
		~MeshResource() = default;
		MeshResource(const MeshResource&) = delete;
		MeshResource(MeshResource&&) = default;
		MeshResource& operator=(const MeshResource&) = delete;
		MeshResource& operator=(MeshResource&&) = default;

	public:
		std::weak_ptr<MeshParam> GetMeshParam() const
		{
			return std::static_pointer_cast<MeshParam>(m_pParam);
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<MeshParam>();

			auto pParam = std::static_pointer_cast<MeshParam>(m_pParam);
			pParam->SetFaceArray(m_face.GetData());
			pParam->SetFaceNormArray(m_faceNorm.GetData());
			pParam->SetFaceAreaArray(m_faceArea);
			pParam->SetNbFacesIdxArray(m_nbFaces);
			pParam->SetNbFacesIdxArray(m_nbFacesIdx);
			pParam->SetNbNodesArray(m_nbNodes);
			pParam->SetNbNodesIdxArray(m_nbNodesIdx);
			pParam->SetRTriangleArray(m_rTri);
			pParam->SetShortestEdgeIdArray(m_shortestEdgeId);
			pParam->SetSamplingParticleSizeArray(m_samplingParticleSize);
			ObjectResource::SetParam();
		}

	private:
		mcuda::gl::DeviceResource<glm::uvec3> m_face;
		mcuda::gl::DeviceResource<REAL3> m_faceNorm;
		REAL* m_faceArea;
		uint32_t* m_nbFaces;
		uint32_t* m_nbFacesIdx;
		uint32_t* m_nbNodes;
		uint32_t* m_nbNodesIdx;
		uint32_t* m_rTri;
		uint32_t* m_shortestEdgeId;
		uint32_t* m_samplingParticleSize;
	};
}