#include "stdafx.h"
#include "KineticToRenderConverter.h"

#include "../MCore_interface/ISimulateManager.h"

#include "../MCore_render/MeshModel.h"
#include "../MCore_render/AABBModel.h"

#include "../MPS_simulate/MeshUtil.h"
#include "../MPS_simulate/DeviceMeshContainer.h"

REGISTRY_SIMULATE_TO_RENDER(KineticToRenderConverter)

void KineticToRenderConverter::OnConvert(SIMULATE_CONTAINER* pSimulateContainer, IRenderModelContainer* pRenderModelManager)
{
	const auto pSimulateManager = pSimulateContainer->GetSimulateManager();
	const auto pMeshContainer = pSimulateManager->GetDeviceContainer<DeviceMeshContainer>();

	const auto meshModel = pRenderModelManager->GetModel<render::MeshModel>();

	//SetupAABBStream(pSimulateContainer, aabbModel);
}
