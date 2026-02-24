#pragma once

#include "DBRegistery.h"

struct PhysicsData : public database::Data
{
	DATABASE_FIELD(PhysicsData, database::Data, (
		((glm::dvec3), gravity, "Gravity"),
		((double), dt, "TimeStep")
	), PHYSICS);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);

		gravity = { 0.0, 0.0, -9.81 };
		dt = 0.01;
	}
};