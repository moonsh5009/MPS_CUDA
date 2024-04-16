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

		void __MY_EXT_CLASS__ ComputeDFSPHFactor(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeDFSPHConstantDensity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeDFSPHDivergenceFree(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

		void __MY_EXT_CLASS__ ApplyExplicitViscosity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyExplicitSurfaceTension(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

		void __MY_EXT_CLASS__ ApplyImplicitViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyImplicitGDViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyImplicitCGViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

		void __MY_EXT_CLASS__ DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterialParam& material);
	}
}

#include "HeaderPost.h"