#pragma once

#include "../MCore_system/SimulateToRenderConverter.h"

#include "../MPS_simulate/DeviceMeshContainer.h"

#include "HeaderPre.h"

namespace mcore::render
{
	class MeshModel;
	class AABBModel;
}

class __MY_EXT_CLASS__ MeshToRenderConverter : public system::SimulateToRenderConverter<DeviceMeshContainer>
{
public:
	void OnConvert(SIMULATE_CONTAINER* pSimulateContainer, IRenderModelContainer* pRenderModelManager) override;

private:
	void SetupTriangleStream(
		const DeviceMeshContainer* pSimulateContainer,
		render::MeshModel* pModel);
	void SetupLineStream(
		const DeviceMeshContainer* pSimulateContainer,
		render::MeshModel* pModel);
	void SetupPointStream(
		const DeviceMeshContainer* pSimulateContainer,
		render::MeshModel* pModel);

	void SetupAABBStream(
		const DeviceMeshContainer* pSimulateContainer,
		render::AABBModel* pModel);
};

#include "HeaderPost.h"