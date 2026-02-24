#pragma once

#include "DBDef.h"

namespace mcore
{
	struct DynamicsContext
	{
		Vector3* meshNodes;
		Vector3* velocities;
		double* masses;
		IndexType* dynNodeIndices;
		IndexType* dynKineticIndices;
		IndexType* isFixed;
		double dt;
		size_t totalNodeCount;
	};

	class IDynamicsContributor
	{
	public:
		virtual ~IDynamicsContributor() = default;

		virtual void AccumulateGradient(const DynamicsContext& ctx, Vector3* rhs) = 0;
		virtual void AccumulateHessianVector(const DynamicsContext& ctx, const Vector3* p, Vector3* Ap) = 0;
		virtual void AccumulateDiagonal(const DynamicsContext& ctx, Vector3* diag) = 0;
		virtual void AccumulateDamping(const DynamicsContext& ctx, Vector3* rhs) {}
	};
}