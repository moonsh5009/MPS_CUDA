#include "stdafx.h"
#include "ClothPool.h"

IMPLEMENT_DATAPOOL(ClothPool, CLOTH)

ClothPool::ClothPool(IDBSession* pDBSession)
	: database::Pool<ClothData>{ pDBSession }
{}