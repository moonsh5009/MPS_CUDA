#pragma once

#include "MPSMaterial.h"

#include "HeaderPre.h"

namespace mps
{
	struct MeshMaterialParam
	{
		REAL radius;
		REAL mass;
		REAL density;

		REAL viscosity;
		REAL surfaceTension;

		glm::fvec4 frontColor;
		glm::fvec4 backColor;
	};
	class __MY_EXT_CLASS__ MeshMaterial : public VirtualMaterial<MeshMaterialParam>
	{
	public:
		MeshMaterial();

	public:
		void SetParam(REAL radius, REAL density);
	};
};

#include "HeaderPost.h"