#pragma once

#include "DBRegistery.h"

struct KineticData : public database::Data
{
	DATABASE_FIELD(KineticData, database::Data, (
		((std::vector<double>), masses, "Masses"),
		((std::vector<double>), invMasses, "InvMasses"),
		((std::vector<mcore::Vector3>), velocities, "Velocities"),
		((std::vector<mcore::Vector3>), forces, "Forces"),
		((std::vector<IndexType>), fixeds, "Fixeds")
	), KINETIC);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);
		masses.clear();
		invMasses.clear();
		velocities.clear();
		forces.clear();
		fixeds.clear();
	}
};