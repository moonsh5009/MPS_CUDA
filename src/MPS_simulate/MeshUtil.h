#pragma once

#include "DeviceMeshContainer.h"

#include "HeaderPre.h"

class __MY_EXT_CLASS__ MeshUtil
{
public:
	static void ComputeNormal(const DeviceMeshData& meshData);
	static void UpdateFaceIndirects(const DeviceMeshData& meshData);
};

#include "HeaderPost.h"
