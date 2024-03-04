#pragma once

#include "MPSModel.h"

#include "MPSSPHMaterial.h"
#include "MPSSPHObject.h"
#include "MPSSpatialHash.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ SPHModel : public Model
	{
	public:
		SPHModel(std::unique_ptr<SPHMaterial>&& pMaterial, std::unique_ptr<SPHObject>&& pObject, std::unique_ptr<SpatialHash>&& pTree);
	};
};

#include "HeaderPost.h"