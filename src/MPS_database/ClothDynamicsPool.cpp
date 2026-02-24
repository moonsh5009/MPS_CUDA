#include "stdafx.h"
#include "ClothDynamicsPool.h"

IMPLEMENT_DATAPOOL(ClothDynamicsPool, CLOTH_DYNAMICS)

ClothDynamicsPool::ClothDynamicsPool(IDBSession* pDBSession)
	: database::Pool<ClothDynamics>{ pDBSession }
{}