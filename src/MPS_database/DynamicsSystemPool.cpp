#include "stdafx.h"
#include "DynamicsSystemPool.h"

IMPLEMENT_DATAPOOL(DynamicsSystemPool, DYNAMICS_SYSTEM)

DynamicsSystemPool::DynamicsSystemPool(IDBSession* pDBSession)
	: database::Pool<DynamicsSystem>{ pDBSession }
{}