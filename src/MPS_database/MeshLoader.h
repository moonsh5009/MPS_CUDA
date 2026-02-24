#pragma once

#include "../MCore_util/AABB.h"

#include "MeshData.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ MeshLoader
{
public:
	MeshLoader(mcore::IDBSession* pSession);

	void LoadOBJ(
		const std::string& filename,
		const std::shared_ptr<MeshData>& pMeshData,
		const AABB<double>& aabb);

private:
	mcore::IDBSession* m_pSession;
};

#include "HeaderPost.h"