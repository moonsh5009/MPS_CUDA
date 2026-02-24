#include "stdafx.h"
#include "MeshPool.h"

IMPLEMENT_DATAPOOL(MeshPool, MESH)

MeshPool::MeshPool(IDBSession* pDBSession)
	: database::Pool<MeshData>{ pDBSession }
{}