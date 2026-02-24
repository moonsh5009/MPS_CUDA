#pragma once

#include "MeshData.h"
#include "KineticData.h"

struct ClothData : public database::Data
{
	DATABASE_FIELD(ClothData, database::Data, (
		((database::Ref<MeshData>), mesh, "Mesh"),
		((database::Ref<KineticData>), kinetic, "Kinetic"),
		((double), stretchStiffness, "StretchStiffness"),
		((double), bendStiffness, "BendStiffness"),
		((double), dampingCoeff, "DampingCoeff"),
		((IndexType), cgMaxIterations, "CGMaxIterations")
	), CLOTH);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);

		mesh.Initialize(pDBSession);
		kinetic.Initialize(pDBSession);
		stretchStiffness = 1000.0;
		bendStiffness = 0.01;
		dampingCoeff = 0.01;
		cgMaxIterations = 30;
	}
};