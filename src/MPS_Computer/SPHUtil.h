#pragma once

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	class SPHParam;
	class BoundaryParticleParam;
	class SpatialHashParam;
	struct SPHMaterialParam;
	struct MeshMaterialParam;
	struct PhysicsParam;
	namespace kernel::sph
	{
		void __MY_EXT_CLASS__ ComputeBoundaryParticleVolume_0(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeBoundaryParticleVolume_1(
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& material, const mps::SpatialHashParam& hash,
			const mps::BoundaryParticleParam& refBoundaryParticle, const mps::MeshMaterialParam& refMaterial, const mps::SpatialHashParam& refHash);
		void __MY_EXT_CLASS__ ComputeBoundaryParticleVolume_2(const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& material);

		void __MY_EXT_CLASS__ ComputeDensity_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeDensity_1(
			const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash);
		void __MY_EXT_CLASS__ ComputeDensity_2(const mps::SPHParam& sph, const mps::SPHMaterialParam& material);

		void __MY_EXT_CLASS__ ComputeDFSPHFactor_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeDFSPHFactor_1(
			const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash);
		void __MY_EXT_CLASS__ ComputeDFSPHFactor_2(const mps::SPHParam& sph, const mps::SPHMaterialParam& material);
		
		void __MY_EXT_CLASS__ ComputeDensityDelta_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ComputeDensityDelta_1(
			const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash);

		void __MY_EXT_CLASS__ ApplyDFSPH_0(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyDFSPH_1(
			const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash);
		void __MY_EXT_CLASS__ ApplyDFSPH_2(const mps::SPHParam& sph, const mps::SPHMaterialParam& material);

		void __MY_EXT_CLASS__ ComputeDFSPHConstantDensity(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash);
		void __MY_EXT_CLASS__ ComputeDFSPHDivergenceFree(const mps::PhysicsParam& physParam,
			const mps::SPHParam& sph, const mps::SPHMaterialParam& sphMaterial, const mps::SpatialHashParam& sphHash,
			const mps::BoundaryParticleParam& boundaryParticle, const mps::MeshMaterialParam& boundaryParticleMaterial, const mps::SpatialHashParam& boundarhParticleHash);

		void __MY_EXT_CLASS__ ApplyExplicitViscosity(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyExplicitSurfaceTension(const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

		void __MY_EXT_CLASS__ ApplyImplicitJacobiViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyImplicitGDViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);
		void __MY_EXT_CLASS__ ApplyImplicitCGViscosity(const mps::PhysicsParam& physParam, const mps::SPHParam& sph, const mps::SPHMaterialParam& material, const mps::SpatialHashParam& hash);

		void __MY_EXT_CLASS__ DensityColorTest(const mps::SPHParam& sph, const mps::SPHMaterialParam& material);
	}
}

#include "HeaderPost.h"