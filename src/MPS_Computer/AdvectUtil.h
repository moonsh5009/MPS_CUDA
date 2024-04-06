#pragma once

#include "../MPS_Object/MPSObject.h"

#include "HeaderPre.h"

namespace mps
{
	class GBArchiver;
	struct PhysicsParam;
	namespace kernel
	{
		__MY_EXT_CLASS__ void InitMass(Object* pObj);
		__MY_EXT_CLASS__ void InitForce(const mps::ObjectParam& objParam);
		__MY_EXT_CLASS__ void ApplyGravity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
		__MY_EXT_CLASS__ void UpdateVelocity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
		__MY_EXT_CLASS__ void UpdatePosition(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam);
	}
}

#include "HeaderPost.h"