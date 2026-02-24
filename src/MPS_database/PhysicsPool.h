#pragma once

#include "../MCore_database/SinglePool.h"

#include "PhysicsData.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ PhysicsPool : public database::SinglePool<PhysicsData>
{
	DECLARE_SINGLE_DATAPOOL(PhysicsPool, PHYSICS)
public:
	PhysicsPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"