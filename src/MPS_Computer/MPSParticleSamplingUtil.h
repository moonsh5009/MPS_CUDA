#pragma once

#include "HeaderPre.h"

namespace mps
{
	struct MeshParam;
	struct MeshMaterialParam;
	struct BoundaryParticleParam;
	class BoundaryParticleObject;
	namespace kernel::ParticleSampling
	{
		bool __MY_EXT_CLASS__ IsSamplingParticleRequired(const MeshMaterialParam& material, const MeshParam& obj, uint32_t* prevIdx, uint32_t* currIdx, bool* isGenerateds);
		void __MY_EXT_CLASS__ ParticleSampling(const MeshMaterialParam& material, const MeshParam& obj, BoundaryParticleObject& boundaryParticle);

		/*void CompNodeWeights(Cloth* cloth, PoreParticle* poreParticles);
		void LerpPosition(MeshObject* obj, BoundaryParticle* boundaryParticles);
		void LerpVelocity(MeshObject* obj, BoundaryParticle* boundaryParticles);
		void LerpForce(Cloth* cloth, PoreParticle* poreParticles);*/
	}
}

#include "HeaderPost.h"