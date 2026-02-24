#pragma once

#include "DynamicsSystem.h"

struct ClothDynamics : public database::Data
{
	DATABASE_FIELD(ClothDynamics, database::Data, (
		((database::Ref<DynamicsSystem>), dynamicsSystem, "DynamicsSystem"),
		((double), stretchStiffness, "StretchStiffness"),
		((double), bendStiffness, "BendStiffness"),
		((double), dampingCoeff, "DampingCoeff")
	), CLOTH_DYNAMICS);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);

		dynamicsSystem.Initialize(pDBSession);
		stretchStiffness = 1000.0;
		bendStiffness = 0.01;
		dampingCoeff = 0.01;
	}
};