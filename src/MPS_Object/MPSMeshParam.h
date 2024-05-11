#pragma once

#include "MPSObjectParam.h"

namespace mps
{
	struct MeshParam : public ObjectParam
	{
		size_t vertexSize;

		glm::uvec3* pFace;
		REAL3* pFaceNorm;
		REAL* pFaceArea;

		uint32_t* pNbFaces;
		uint32_t* pNbFacesIdx;
		uint32_t* pNbNodes;
		uint32_t* pNbNodesIdx;

		uint32_t* pRTri;
		uint32_t* pShortestEdgeID;
		uint32_t* pSamplingParticleSize;
	};

	struct MeshResource : public ObjectResource
	{
	public:
		MeshResource() = delete;
		MeshResource(std::shared_ptr<ObjectResource> pSuper, size_t vertexSize,
			mcuda::gl::DeviceResource<glm::uvec3>&& face, mcuda::gl::DeviceResource<REAL3>&& faceNorm,
			REAL* faceArea, uint32_t* nbFaces, uint32_t* nbFacesIdx, uint32_t* nbNodes, uint32_t* nbNodesIdx,
			uint32_t* rTri, uint32_t* shortestEdgeId, uint32_t* samplingParticleSize) :
			ObjectResource{ std::move(*pSuper) }, m_vertexSize{ vertexSize },
			m_face { std::move(face) }, m_faceNorm{ std::move(faceNorm) },
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
			pParam->pFace = m_face.GetData();
			pParam->pFaceNorm = m_faceNorm.GetData();
			pParam->pFaceArea = m_faceArea;
			pParam->pNbFaces = m_nbFaces;
			pParam->pNbFacesIdx = m_nbFacesIdx;
			pParam->pNbNodes = m_nbNodes;
			pParam->pNbNodesIdx = m_nbNodesIdx;
			pParam->pRTri = m_rTri;
			pParam->pShortestEdgeID = m_shortestEdgeId;
			pParam->pSamplingParticleSize = m_samplingParticleSize;
			ObjectResource::SetParam();
		}

	private:
		size_t m_vertexSize;
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