#include "stdafx.h"
#include "KineticPool.h"

IMPLEMENT_DATAPOOL(KineticPool, KINETIC)

KineticPool::KineticPool(IDBSession* pDBSession)
	: database::Pool<KineticData>{ pDBSession }
{}