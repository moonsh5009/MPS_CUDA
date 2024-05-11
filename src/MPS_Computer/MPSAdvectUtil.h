#pragma once

#include "../MPS_Object/MPSDef.h"

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	struct ObjectParam;
	struct PhysicsParam;
	namespace kernel::Advect
	{
		void __MY_EXT_CLASS__ ResetREAL(REAL* ptr, size_t size);
		void __MY_EXT_CLASS__ ResetForce(const mps::ObjectParam& obj);
		void __MY_EXT_CLASS__ ApplyGravity(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj);
		void __MY_EXT_CLASS__ UpdateVelocity(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj);
		void __MY_EXT_CLASS__ UpdatePosition(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj);
		void __MY_EXT_CLASS__ BoundaryCollision(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj);
	}
}

#include "HeaderPost.h"