#pragma once

#include "../MCore_database/Pool.h"
#include "ClothData.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ ClothPool : public database::Pool<ClothData>
{
	DECLARE_DATAPOOL(ClothPool, CLOTH)
public:
	ClothPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"