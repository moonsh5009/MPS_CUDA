#pragma once

#include "MPSSPHUtilDef.h"

#include "HeaderPre.h"

namespace mps::kernel::SPH
{
	/*void __MY_EXT_CLASS__ ApplyExplicitViscosity(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ApplyExplicitSurfaceTension(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);

	void __MY_EXT_CLASS__ ApplyImplicitJacobiViscosity(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ApplyImplicitCGViscosity(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);*/

	void __MY_EXT_CLASS__ ComputeSurfaceTensionFactor(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ComputeSurfaceTensionCGb(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ComputeSurfaceTensionCGAp(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		REAL3* pCGp,
		REAL3* pCGAp);
	void __MY_EXT_CLASS__ ComputeViscosityCGAp(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		REAL3* pCGp,
		REAL3* pCGAp);
	void __MY_EXT_CLASS__ ComputeFinalCGAp(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		REAL3* pCGp,
		REAL3* pCGAp,
		REAL factor);

	void __MY_EXT_CLASS__ ApplyImplicitViscosityNSurfaceTension(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
}

#include "HeaderPost.h"