#pragma once

#include "../MCore_database/Pool.h"
#include "DynamicsSystem.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ DynamicsSystemPool : public database::Pool<DynamicsSystem>
{
	DECLARE_DATAPOOL(DynamicsSystemPool, DYNAMICS_SYSTEM)
public:
	DynamicsSystemPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"