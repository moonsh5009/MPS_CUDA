#pragma once

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	class Object;
	class ObjectParam;
	struct PhysicsParam;
	namespace kernel
	{
		void __MY_EXT_CLASS__ InitMass(Object* pObj);
		void __MY_EXT_CLASS__ ResetForce(const mps::ObjectParam& objParam);
		void __MY_EXT_CLASS__ ApplyGravity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
		void __MY_EXT_CLASS__ UpdateVelocity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
		void __MY_EXT_CLASS__ UpdatePosition(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
		void __MY_EXT_CLASS__ BoundaryCollision(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
	}
}

#include "HeaderPost.h"