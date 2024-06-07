#pragma once

#include "MPSSPHUtilDef.h"

#include "HeaderPre.h"

namespace mps::kernel::SPH
{
	void __MY_EXT_CLASS__ ComputeDFSPHFactorSub(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ComputeDFSPHFactorSub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::SPHMaterialParam& refSPHMaterial,
		const mps::SPHParam& refSPH);
	void __MY_EXT_CLASS__ ComputeDFSPHFactorSub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::BoundaryParticleParam& boundaryParticle);
	void __MY_EXT_CLASS__ ComputeDFSPHFactorFinal(
		const mps::SPHParam& sph);

	void __MY_EXT_CLASS__ ComputeDensityDeltaSub(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		long long stream = 0);
	void __MY_EXT_CLASS__ ComputeDensityDeltaSub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::SPHMaterialParam& refSPHMaterial,
		const mps::SPHParam& refSPH,
		long long stream = 0);
	void __MY_EXT_CLASS__ ComputeDensityDeltaSub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::BoundaryParticleParam& boundaryParticle,
		long long stream = 0);

	void __MY_EXT_CLASS__ ComputeDFSPHConstantDensitySub(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		REAL* sumError);
	void __MY_EXT_CLASS__ ComputeDFSPHDivergenceFreeSub(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		REAL* sumError);

	void __MY_EXT_CLASS__ ApplyDFSPHSub(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		long long stream = 0);
	void __MY_EXT_CLASS__ ApplyDFSPHSub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::SPHMaterialParam& pRefSPHMaterial,
		const mps::SPHParam& refSPH,
		long long stream = 0);
	void __MY_EXT_CLASS__ ApplyDFSPHSub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::BoundaryParticleParam& boundaryParticle,
		long long stream = 0);
	void __MY_EXT_CLASS__ ApplyDFSPHFinal(
		const mps::SPHParam& sph);

	void __MY_EXT_CLASS__ ComputeDFSPHConstantDensity(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::MeshMaterialParam& boundaryParticleMaterial,
		const mps::BoundaryParticleParam& boundaryParticle,
		const mps::NeiParam& neiSPH2SPH,
		const mps::NeiParam& neiSPH2BoundaryParticle);
	void __MY_EXT_CLASS__ ComputeDFSPHDivergenceFree(
		const mps::PhysicsParam& physParam,
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::MeshMaterialParam& boundaryParticleMaterial,
		const mps::BoundaryParticleParam& boundaryParticle,
		const mps::NeiParam& neiSPH2SPH,
		const mps::NeiParam& neiSPH2BoundaryParticle);
}

#include "HeaderPost.h"