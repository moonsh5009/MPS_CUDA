#pragma once

#include "MPSMaterial.h"

#include "HeaderPre.h"

namespace mps
{
	struct SPHMaterialParam
	{
		size_t numParticles;
		REAL radius;
		REAL mass;
		REAL density;
		glm::fvec4 color;
	};
	class __MY_EXT_CLASS__ SPHMaterial : public VirtualMaterial<SPHMaterialParam>
	{
	public:
		SPHMaterial();

	public:
		void SetParticleSize(const uint32_t size);
		void SetRadius(const REAL radius);
		void SetMass(const REAL mass);
		void SetDensity(const REAL density);
		void SetColor(const glm::fvec4& color);

		constexpr uint32_t GetParticleSize() const { return GetHost().numParticles; };
		constexpr REAL GetRadius() const { return GetHost().radius; };
		constexpr REAL GetMass() const { return GetHost().mass; };
		constexpr REAL GetDensity() const { return GetHost().density; };
		constexpr const glm::fvec4& GetColor() const { return GetHost().color; };
	};
};

#include "HeaderPost.h"