#pragma once

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	class SPHParam;
	class SpatialHashParam;
	struct SPHMaterialParam;
	struct PhysicsParam;
	namespace kernel::sph
	{
		void __MY_EXT_CLASS__ ComputeDensity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterialParam& material);

		void __MY_EXT_CLASS__ ComputeDFSPHFactor(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputePressureForce(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeDivergenceFree(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyDFSPH(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

		void __MY_EXT_CLASS__ ComputeColorField(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplySurfaceTension(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

	}
}

#include "HeaderPost.h"