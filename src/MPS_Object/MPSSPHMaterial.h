#pragma once

#include "MPSMaterial.h"

#include "HeaderPre.h"

namespace mps
{
	struct SPHMaterialParam
	{
		REAL radius;
		REAL mass;
		REAL density;
		REAL volume;

		REAL viscosity;
		REAL surfaceTension;

		glm::fvec4 color;
	};
	class __MY_EXT_CLASS__ SPHMaterial : public VirtualMaterial<SPHMaterialParam>
	{
	public:
		SPHMaterial();

	public:
		constexpr void SetParam(REAL radius, REAL density);
		constexpr void SetRadius(REAL radius);
		constexpr void SetViscosity(REAL viscosity);
		constexpr void SetSurfaceTension(REAL surfaceTension);
		void SetColor(const glm::fvec4& color);

		constexpr REAL GetRadius() const { return GetParam().radius; };
		constexpr REAL GetMass() const { return GetParam().mass; };
		constexpr REAL GetDensity() const { return GetParam().density; };
		constexpr REAL GetVolume() const { return GetParam().volume; };
		constexpr REAL GetViscosity() const { return GetParam().viscosity; };
		constexpr REAL GetSurfaceTension() const { return GetParam().surfaceTension; };
		constexpr const glm::fvec4& GetColor() const { return GetParam().color; };
	};
};

#include "HeaderPost.h"