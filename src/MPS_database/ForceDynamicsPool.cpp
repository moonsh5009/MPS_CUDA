#include "stdafx.h"
#include "ForceDynamicsPool.h"

IMPLEMENT_DATAPOOL(ForceDynamicsPool, FORCE_DYNAMICS)

ForceDynamicsPool::ForceDynamicsPool(IDBSession* pDBSession)
	: database::Pool<ForceDynamics>{ pDBSession }
{}