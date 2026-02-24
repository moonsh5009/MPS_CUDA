#pragma once

#include "../MCore_database/Pool.h"
#include "KineticData.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ KineticPool : public database::Pool<KineticData>
{
	DECLARE_DATAPOOL(KineticPool, KINETIC)
public:
	KineticPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"