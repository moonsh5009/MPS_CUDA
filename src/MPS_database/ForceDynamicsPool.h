#pragma once

#include "../MCore_database/Pool.h"

#include "ForceDynamics.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ ForceDynamicsPool : public database::Pool<ForceDynamics>
{
	DECLARE_DATAPOOL(ForceDynamicsPool, FORCE_DYNAMICS)
public:
	ForceDynamicsPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"