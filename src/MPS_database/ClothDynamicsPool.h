#pragma once

#include "../MCore_database/Pool.h"
#include "ClothDynamics.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ ClothDynamicsPool : public database::Pool<ClothDynamics>
{
	DECLARE_DATAPOOL(ClothDynamicsPool, CLOTH_DYNAMICS)
public:
	ClothDynamicsPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"