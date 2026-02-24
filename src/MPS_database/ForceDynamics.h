#pragma once

#include "MeshData.h"
#include "KineticData.h"

struct ForceDynamics : public database::Data
{
	DATABASE_FIELD(ForceDynamics, database::Data, (
		((database::Ref<MeshData>), mesh, "Mesh"),
		((database::Ref<KineticData>), kinetic, "Kinetic")
	), FORCE_DYNAMICS);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);

		mesh.Initialize(pDBSession);
		kinetic.Initialize(pDBSession);
	}
};