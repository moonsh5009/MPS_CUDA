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
		REAL pressureAtm;
		glm::fvec4 color;
	};
	class __MY_EXT_CLASS__ SPHMaterial : public VirtualMaterial<SPHMaterialParam>
	{
	public:
		SPHMaterial();

	public:
		void SetParam(const REAL radius, const REAL density);
		void SetRadius(const REAL radius);
		void SetViscosity(const REAL viscosity);
		void SetSurfaceTension(const REAL surfaceTension);
		void SetPressureAtm(const REAL pressureAtm);
		void SetColor(const glm::fvec4& color);

		REAL GetRadius() const { return GetParam().radius; };
		REAL GetMass() const { return GetParam().mass; };
		REAL GetDensity() const { return GetParam().density; };
		REAL GetVolume() const { return GetParam().volume; };
		REAL GetViscosity() const { return GetParam().viscosity; };
		REAL GetSurfaceTension() const { return GetParam().surfaceTension; };
		REAL GetPressureAtm() const { return GetParam().pressureAtm; };
		const glm::fvec4& GetColor() const { return GetParam().color; };
	};
};

#include "HeaderPost.h"