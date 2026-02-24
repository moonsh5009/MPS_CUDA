#pragma once

#include "../MCore_database/Pool.h"

#include "MeshData.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ MeshPool : public database::Pool<MeshData>
{
	DECLARE_DATAPOOL(MeshPool, MESH)
public:
	MeshPool(IDBSession* pDBSession);
};

#include "HeaderPost.h"