#pragma once

#include "../MCore_simulate/DeviceConstant.h"

#include "../MPS_database/PhysicsData.h"

#include "HeaderPre.h"

struct DevicePhysicsData
{
	glm::dvec3 gravity;
	double dt;
};

class __MY_EXT_CLASS__ DevicePhysicsConstant : public simulate::DeviceConstant<PhysicsData>
{
public:
	DevicePhysicsConstant(ISimulateManager* pSimulateManager);

	void Initialize() override;
	void OnModifyDB(const DB* prevDB, const DB* newDB) override;
};

#include "HeaderPost.h"
