#pragma once

#include "MPSMaterial.h"

#include "HeaderPre.h"

namespace mps
{
	struct SPHMaterialParam
	{
		size_t numParticles;
		float radius;
		float mass;
		float density;
		glm::fvec4 color;
	};
	class __MY_EXT_CLASS__ SPHMaterial : public VirtualMaterial<SPHMaterialParam>
	{
	public:
		SPHMaterial();

	public:
		void SetParticleSize(const uint32_t size);
		void SetRadius(const float radius);
		void SetMass(const float mass);
		void SetDensity(const float density);
		void SetColor( const glm::fvec4& color);

		constexpr uint32_t GetParticleSize() const { return GetHost().numParticles; };
		constexpr float GetRadius() const { return GetHost().radius; };
		constexpr float GetMass() const { return GetHost().mass; };
		constexpr float GetDensity() const { return GetHost().density; };
		constexpr const glm::fvec4& GetColor() const { return GetHost().color; };
	};
};

#include "HeaderPost.h"