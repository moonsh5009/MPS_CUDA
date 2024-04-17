#pragma once

#include "MPSObjectParam.h"

namespace mps
{
	class MeshParam : public ObjectParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC MeshParam() : ObjectParam{}, m_pFace{ nullptr }, m_pNbFaces{ nullptr }, m_pIdxNbFaces{ nullptr }, m_pNbNodes{ nullptr }, m_pIdxNbNodes{ nullptr } {};
		MCUDA_HOST_DEVICE_FUNC ~MeshParam() {}

	public:
		MCUDA_DEVICE_FUNC glm::uvec3& Face(uint32_t idx) { return m_pFace[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& NbFaces(uint32_t idx) { return m_pNbFaces[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& IdxNbFaces(uint32_t idx) { return m_pIdxNbFaces[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& NbNodes(uint32_t idx) { return m_pNbNodes[idx]; }
		MCUDA_DEVICE_FUNC uint32_t& IdxNbNodes(uint32_t idx) { return m_pIdxNbNodes[idx]; }

		MCUDA_DEVICE_FUNC const glm::uvec3 Face(uint32_t idx) const { return m_pFace[idx]; }
		MCUDA_DEVICE_FUNC const uint32_t NbFaces(uint32_t idx) const { return m_pNbFaces[idx]; }
		MCUDA_DEVICE_FUNC const uint32_t IdxNbFaces(uint32_t idx) const { return m_pIdxNbFaces[idx]; }
		MCUDA_DEVICE_FUNC const uint32_t NbNodes(uint32_t idx) const { return m_pNbNodes[idx]; }
		MCUDA_DEVICE_FUNC const uint32_t IdxNbNodes(uint32_t idx) const { return m_pIdxNbNodes[idx]; }

	public:
		MCUDA_HOST_FUNC glm::uvec3* GetFaceArray() const { return m_pFace; }
		MCUDA_HOST_FUNC uint32_t* GetNbFacesArray() const { return m_pNbFaces; }
		MCUDA_HOST_FUNC uint32_t* GetIdxNbFacesArray() const { return m_pIdxNbFaces; }
		MCUDA_HOST_FUNC uint32_t* GetNbNodesArray() const { return m_pNbNodes; }
		MCUDA_HOST_FUNC uint32_t* GetIdxNbNodesArray() const { return m_pIdxNbNodes; }

		MCUDA_HOST_FUNC void SetFaceArray(glm::uvec3* pFace) { m_pFace = pFace; }
		MCUDA_HOST_FUNC void SetNbFacesArray(uint32_t* pNbFaces) { m_pNbFaces = pNbFaces; }
		MCUDA_HOST_FUNC void SetIdxNbFacesArray(uint32_t* pIdxNbFaces) { m_pIdxNbFaces = pIdxNbFaces; }
		MCUDA_HOST_FUNC void SetNbNodesArray(uint32_t* pNbNodes) { m_pNbNodes = pNbNodes; }
		MCUDA_HOST_FUNC void SetIdxNbNodesArray(uint32_t* pIdxNbNodes) { m_pIdxNbNodes = pIdxNbNodes; }

	private:
		glm::uvec3* MCUDA_RESTRICT m_pFace;
		uint32_t* MCUDA_RESTRICT m_pNbFaces;
		uint32_t* MCUDA_RESTRICT m_pIdxNbFaces;
		uint32_t* MCUDA_RESTRICT m_pNbNodes;
		uint32_t* MCUDA_RESTRICT m_pIdxNbNodes;
	};

	struct MeshResource : public ObjectResource
	{
	public:
		MeshResource() = delete;
		MeshResource(std::shared_ptr<ObjectResource> pSuper, mcuda::gl::DeviceResource<glm::uvec3>&& face,
			uint32_t* nbFaces, uint32_t* idxNbFaces, uint32_t* nbNodes, uint32_t* idxNbNodes) :
			ObjectResource{ std::move(*pSuper) }, m_face{ std::move(face) }, m_nbFaces{ nbFaces }, m_idxNbFaces{ idxNbFaces }, m_nbNodes{ nbNodes }, m_idxNbNodes{ idxNbNodes }
		{}
		~MeshResource() = default;
		MeshResource(const MeshResource&) = delete;
		MeshResource(MeshResource&&) = default;
		MeshResource& operator=(const MeshResource&) = delete;
		MeshResource& operator=(MeshResource&&) = default;

	public:
		std::weak_ptr<MeshParam> GetParticleParam() const
		{
			return std::static_pointer_cast<MeshParam>(m_pParam);
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<MeshParam>();

			auto pParam = std::static_pointer_cast<MeshParam>(m_pParam);
			pParam->SetFaceArray(m_face.GetData());
			pParam->SetIdxNbFacesArray(m_nbFaces);
			pParam->SetIdxNbFacesArray(m_idxNbFaces);
			pParam->SetNbNodesArray(m_nbNodes);
			pParam->SetIdxNbNodesArray(m_idxNbNodes);
			ObjectResource::SetParam();
		}

	private:
		mcuda::gl::DeviceResource<glm::uvec3> m_face;
		uint32_t* m_nbFaces;
		uint32_t* m_idxNbFaces;
		uint32_t* m_nbNodes;
		uint32_t* m_idxNbNodes;
	};
}