#pragma once

#include "HeaderPre.h"

namespace mps
{
	class MeshParam;
	struct MeshMaterialParam;
	class BoundaryParticleParam;
	class BoundaryParticleObject;
	namespace kernel::ParticleSampling
	{
		bool __MY_EXT_CLASS__ IsSamplingParticleRequired(const MeshParam& obj, const MeshMaterialParam& material, uint32_t* prevIdx, uint32_t* currIdx, bool* isGenerateds);
		void __MY_EXT_CLASS__ ParticleSampling(const MeshParam& obj, const MeshMaterialParam& material, BoundaryParticleObject& boundaryParticle);

		/*void CompNodeWeights(Cloth* cloth, PoreParticle* poreParticles);
		void LerpPosition(MeshObject* obj, BoundaryParticle* boundaryParticles);
		void LerpVelocity(MeshObject* obj, BoundaryParticle* boundaryParticles);
		void LerpForce(Cloth* cloth, PoreParticle* poreParticles);*/
	}
}

#include "HeaderPost.h"