#pragma once

#include "../MCore_system/SimulateToRenderConverter.h"

#include "../MPS_simulate/DeviceKineticContainer.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ KineticToRenderConverter : public system::SimulateToRenderConverter<DeviceKineticContainer>
{
public:
	void OnConvert(SIMULATE_CONTAINER* pSimulateContainer, IRenderModelContainer* pRenderModelManager) override;
};

#include "HeaderPost.h"