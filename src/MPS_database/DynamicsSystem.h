#pragma once

#include "MeshData.h"
#include "KineticData.h"

struct DynamicsSystem : public database::Data
{
	DATABASE_FIELD(DynamicsSystem, database::Data, (
		((database::Ref<MeshData>), mesh, "Mesh"),
		((database::Ref<KineticData>), kinetic, "Kinetic"),
		((IndexType), maxIterations, "MaxIterations"),
		((double), tolerance, "Tolerance")
	), DYNAMICS_SYSTEM);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);

		mesh.Initialize(pDBSession);
		kinetic.Initialize(pDBSession);
		maxIterations = 30;
		tolerance = 1e-8;
	}
};