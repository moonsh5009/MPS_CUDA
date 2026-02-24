#include "stdafx.h"
#include "PhysicsPool.h"

IMPLEMENT_SINGLE_DATAPOOL(PhysicsPool, PHYSICS)

PhysicsPool::PhysicsPool(IDBSession* pDBSession)
	: database::SinglePool<PhysicsData>{ pDBSession }
{}