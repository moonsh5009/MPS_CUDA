#pragma once

#include "../MCore_simulate/SimulateManager.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ PhysicsSimulateManager : public mcore::simulate::SimulateManager
{
public:
	void Initialize() override;
};

#include "HeaderPost.h"