#pragma once

#include "MPSSPHUtilDef.h"

#include "HeaderPre.h"

namespace mps::kernel::SPH
{
	void __MY_EXT_CLASS__ ComputeBoundaryParticleVolumeSub(
		const mps::BoundaryParticleParam& boundaryParticle,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ComputeBoundaryParticleVolumeSub(
		const mps::BoundaryParticleParam& boundaryParticle,
		const mps::NeiParam& nei,
		const mps::BoundaryParticleParam& refBoundaryParticle);
	void __MY_EXT_CLASS__ ComputeBoundaryParticleVolumeFinal(
		const mps::BoundaryParticleParam& boundaryParticle);

	void __MY_EXT_CLASS__ ComputeDensitySub(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph,
		const mps::NeiParam& nei);
	void __MY_EXT_CLASS__ ComputeDensitySub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::SPHMaterialParam& refSPHMaterial,
		const mps::SPHParam& refSPH);
	void __MY_EXT_CLASS__ ComputeDensitySub(
		const mps::SPHParam& sph,
		const mps::NeiParam& nei,
		const mps::BoundaryParticleParam& boundaryParticle);
	void __MY_EXT_CLASS__ ComputeDensityFinal(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph);

	void __MY_EXT_CLASS__ DensityColorTest(
		const mps::SPHMaterialParam& sphMaterial,
		const mps::SPHParam& sph);
}

#include "HeaderPost.h"