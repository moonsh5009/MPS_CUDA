#include "stdafx.h"
#include "MeshToRenderConverter.h"

#include "../MCore_render/MeshModel.h"
#include "../MCore_render/AABBModel.h"
#include "../MCore_render/RenderCore.h"

#include "../MPS_simulate/MeshUtil.h"

REGISTRY_SIMULATE_TO_RENDER(MeshToRenderConverter)

void MeshToRenderConverter::OnConvert(SIMULATE_CONTAINER* pSimulateContainer, IRenderModelContainer* pRenderModelManager)
{
	const auto meshModel = pRenderModelManager->GetModel<render::MeshModel>();
	const auto aabbModel = pRenderModelManager->GetModel<render::AABBModel>();
	pRenderModelManager->GetRenderCore()->SetZoomFitAllScenes();

	SetupTriangleStream(pSimulateContainer, meshModel);
	SetupLineStream(pSimulateContainer, meshModel);
	SetupPointStream(pSimulateContainer, meshModel);

	SetupAABBStream(pSimulateContainer, aabbModel);
}

void MeshToRenderConverter::SetupTriangleStream(
	const DeviceMeshContainer* pSimulateContainer,
	render::MeshModel* pModel)
{
	if (pSimulateContainer->faceIndices.IsEmpty())
	{
		pModel->GetTriangleStream().SetIBO(nullptr);
		pModel->GetTriangleStream().SetPosition(nullptr);
		pModel->GetTriangleStream().SetNormal(nullptr);
		pModel->GetTriangleStream().SetAttribute(nullptr);
		pModel->GetTriangleStream().SetIndirect(nullptr);
		return;
	}

	const auto& indices = pSimulateContainer->faceIndices.GetBuffer();
	const auto& positions = pSimulateContainer->GetNodes();
	const auto& normals = pSimulateContainer->GetNormals();
	const auto& attributes = pSimulateContainer->GetTriangleAttributes();
	const auto& indirects = pSimulateContainer->faceDrawIndirects.GetBuffer();

	pModel->GetTriangleStream().SetIBO(&indices);
	pModel->GetTriangleStream().SetPosition(&positions);
	pModel->GetTriangleStream().SetNormal(&normals);
	pModel->GetTriangleStream().SetAttribute(&attributes);
	pModel->GetTriangleStream().SetIndirect(&indirects);

	auto meshData = pSimulateContainer->GetDeviceData();
	MeshUtil::ComputeNormal(meshData);
	MeshUtil::UpdateFaceIndirects(meshData);
}

void MeshToRenderConverter::SetupLineStream(
	const DeviceMeshContainer* pSimulateContainer,
	render::MeshModel* pModel)
{
	if (pSimulateContainer->edgeIndices.IsEmpty())
	{
		pModel->GetLineStream().SetIBO(nullptr);
		pModel->GetLineStream().SetPosition(nullptr);
		pModel->GetLineStream().SetAttribute(nullptr);
		pModel->GetLineStream().SetVertexOffset(nullptr);
		pModel->GetLineStream().SetIndirect(nullptr);
		return;
	}

	const auto& indices = pSimulateContainer->edgeIndices.GetBuffer();
	const auto& positions = pSimulateContainer->GetNodes();
	const auto& attributes = pSimulateContainer->GetLineAttributes();
	const auto& vertexOffsets = pSimulateContainer->nodeBuffers.GetOffsets();
	const auto& indirects = pSimulateContainer->edgeDrawIndirects.GetBuffer();

	pModel->GetLineStream().SetIBO(&indices);
	pModel->GetLineStream().SetPosition(&positions);
	pModel->GetLineStream().SetAttribute(&attributes);
	pModel->GetLineStream().SetVertexOffset(&vertexOffsets);
	pModel->GetLineStream().SetIndirect(&indirects);

	std::vector<mvk::DrawIndirectCommand> hostIndirects(pSimulateContainer->edgeIndices.GetSize());
	for (size_t i = 0; i < pSimulateContainer->edgeIndices.GetRanges().size(); ++i)
	{
		hostIndirects[i].vertexCount = static_cast<uint32_t>(pSimulateContainer->edgeIndices.GetRanges()[i].size * 3);
		hostIndirects[i].instanceCount = 1;
		hostIndirects[i].firstVertex = static_cast<uint32_t>(pSimulateContainer->edgeIndices.GetRanges()[i].offset * 3);
		hostIndirects[i].firstInstance = static_cast<uint32_t>(i);
	}
	indirects.CopyFromHost(hostIndirects.data(), indirects.GetElementOffset(hostIndirects.size()));
}

void MeshToRenderConverter::SetupPointStream(
	const DeviceMeshContainer* pSimulateContainer,
	render::MeshModel* pModel)
{
	if (pSimulateContainer->nodeIndices.IsEmpty())
	{
		pModel->GetPointStream().SetIBO(nullptr);
		pModel->GetPointStream().SetPosition(nullptr);
		pModel->GetPointStream().SetAttribute(nullptr);
		pModel->GetPointStream().SetVertexOffset(nullptr);
		pModel->GetPointStream().SetIndirect(nullptr);
		return;
	}

	const auto& indices = pSimulateContainer->nodeIndices.GetBuffer();
	const auto& positions = pSimulateContainer->GetNodes();
	const auto& attributes = pSimulateContainer->GetPointAttributes();
	const auto& vertexOffsets = pSimulateContainer->nodeBuffers.GetOffsets();
	const auto& indirects = pSimulateContainer->nodeDrawIndirects.GetBuffer();

	pModel->GetPointStream().SetIBO(&indices);
	pModel->GetPointStream().SetPosition(&positions);
	pModel->GetPointStream().SetAttribute(&attributes);
	pModel->GetPointStream().SetVertexOffset(&vertexOffsets);
	pModel->GetPointStream().SetIndirect(&indirects);

	std::vector<mvk::DrawIndirectCommand> hostIndirects(pSimulateContainer->nodeIndices.GetSize());
	for (size_t i = 0; i < pSimulateContainer->nodeIndices.GetRanges().size(); ++i)
	{
		hostIndirects[i].vertexCount = static_cast<uint32_t>(pSimulateContainer->nodeIndices.GetRanges()[i].size * 3);
		hostIndirects[i].instanceCount = 1;
		hostIndirects[i].firstVertex = static_cast<uint32_t>(pSimulateContainer->nodeIndices.GetRanges()[i].offset * 3);
		hostIndirects[i].firstInstance = static_cast<uint32_t>(i);
	}
	indirects.CopyFromHost(hostIndirects.data(), indirects.GetElementOffset(hostIndirects.size()));
}

void MeshToRenderConverter::SetupAABBStream(
	const DeviceMeshContainer* pSimulateContainer,
	render::AABBModel* pModel)
{
	if (pSimulateContainer->bvhTree->GetRenderLineIndices().IsEmpty())
	{
		pModel->GetLineStream().SetIBO(nullptr);
		pModel->GetLineStream().SetPosition(nullptr);
		pModel->GetLineStream().SetAttribute(nullptr);
		pModel->GetLineStream().SetVertexOffset(nullptr);
		pModel->GetLineStream().SetIndirect(nullptr);
		return;
	}

	const auto& indices = pSimulateContainer->bvhTree->GetRenderLineIndices();
	const auto& positions = pSimulateContainer->bvhTree->GetRenderVertexPositions();
	const auto& vertexOffsets = pSimulateContainer->bvhTree->GetRenderVertexOffsets();
	const auto& indirects = pSimulateContainer->bvhTree->GetRenderLineDrawIndirects();

	pSimulateContainer->bvhTree->SetTotalAABBUpdateFunction([pModel](const mcore::AABB<REAL>& aabb)
	{
		/*Logger::Info("Total AABB after refit: min(",
			totalAABB.GetMin().x, ", ",
			totalAABB.GetMin().y, ", ",
			totalAABB.GetMin().z, "), max(",
			totalAABB.GetMax().x, ", ",
			totalAABB.GetMax().y, ", ",
			totalAABB.GetMax().z, ")");
		Logger::Print();*/

		AABBf aabbf;
		aabbf.GetMin() = aabb.GetMin();
		aabbf.GetMax() = aabb.GetMax();
		pModel->SetAABB(aabbf);
	});

	pModel->GetLineStream().SetIBO(&indices);
	pModel->GetLineStream().SetPosition(&positions);
	pModel->GetLineStream().SetAttribute(nullptr);
	pModel->GetLineStream().SetVertexOffset(&vertexOffsets);
	pModel->GetLineStream().SetIndirect(&indirects);
}
