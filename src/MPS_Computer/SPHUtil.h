#pragma once

#include "HeaderPre.h"

namespace mps
{
	struct PhysicsParam;

	class SPHParam;
	class BoundaryParticleParam;

	class SpatialHash;
	class SPHMaterial;
	class MeshMaterial;

	namespace kernel::sph
	{
		void __MY_EXT_CLASS__ ComputeBoundaryParticleVolume_0(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ComputeBoundaryParticleVolume_1(
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pMaterial,
			const mps::BoundaryParticleParam& refBoundaryParticle, const mps::MeshMaterial* pRefMaterial,
			const mps::SpatialHash* pRefHash);
		void __MY_EXT_CLASS__ ComputeBoundaryParticleVolume_2(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pMaterial);

		void __MY_EXT_CLASS__ ComputeDensity_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ComputeDensity_1(
			const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
			const mps::SpatialHash* pSPHHash);
		void __MY_EXT_CLASS__ ComputeDensity_2(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial);

		void __MY_EXT_CLASS__ ComputeDFSPHFactor_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ComputeDFSPHFactor_1(
			const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
			const mps::SpatialHash* pSPHHash);
		void __MY_EXT_CLASS__ ComputeDFSPHFactor_2(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial);
		
		void __MY_EXT_CLASS__ ComputeDensityDelta_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ComputeDensityDelta_1(
			const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
			const mps::SpatialHash* pSPHHash);

		void __MY_EXT_CLASS__ ApplyDFSPH_0(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ApplyDFSPH_1(
			const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
			const mps::SpatialHash* pSPHHash);
		void __MY_EXT_CLASS__ ApplyDFSPH_2(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial);

		void __MY_EXT_CLASS__ ComputeDFSPHConstantDensity(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
			const mps::SpatialHash* pSPHHash);
		void __MY_EXT_CLASS__ ComputeDFSPHDivergenceFree(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterial* pSPHMaterial,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterial* pBoundaryParticleMaterial,
			const mps::SpatialHash* pSPHHash);

		void __MY_EXT_CLASS__ ApplyExplicitViscosity(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ApplyExplicitSurfaceTension(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);

		void __MY_EXT_CLASS__ ApplyImplicitJacobiViscosity(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ApplyImplicitGDViscosity(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);
		void __MY_EXT_CLASS__ ApplyImplicitCGViscosity(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial, const mps::SpatialHash* pHash);

		void __MY_EXT_CLASS__ DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterial* pMaterial);
	}
}

#include "HeaderPost.h"