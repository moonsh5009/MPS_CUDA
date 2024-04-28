#pragma once

#include "MPSModel.h"

#include "MPSSPHObject.h"
#include "MPSSPHMaterial.h"
#include "MPSSpatialHash.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ SPHModel : public VirtualModel<0u, 1u, 1u>
	{
		using SuperClass = VirtualModel<0u, 1u, 1u>;
	public:
		SPHModel(std::unique_ptr<SPHObject>&& pObject,
			std::unique_ptr<SPHMaterial>&& pMaterial,
			std::unique_ptr<SpatialHash>&& pTree);
	};
};

#include "HeaderPost.h"