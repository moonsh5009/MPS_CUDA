#include "stdafx.h"
#include "MPSSPHMaterial.h"

mps::SPHMaterial::SPHMaterial() : mps::VirtualMaterial<mps::SPHMaterialParam>{}
{
	SetParam(0.01, 1.0);
	SetViscosity(0.0125);
	SetSurfaceTension(70.0);
	SetColor({ 0.3f, 0.8f, 0.2f, 1.0f });
}

constexpr void mps::SPHMaterial::SetParam(REAL radius, REAL density)
{
	GetParam().radius = radius;
	GetParam().volume = radius * radius * radius / 48.0 * 3.141592;
	GetParam().density = density;
	GetParam().mass = density * GetParam().volume;
}

constexpr void mps::SPHMaterial::SetRadius(REAL radius)
{
	SetParam(radius, GetDensity());
}

constexpr void mps::SPHMaterial::SetViscosity(REAL viscosity)
{
	GetParam().viscosity = viscosity;
}

constexpr void mps::SPHMaterial::SetSurfaceTension(REAL surfaceTension)
{
	GetParam().surfaceTension = surfaceTension;
}

void mps::SPHMaterial::SetColor(const glm::fvec4& color)
{
	GetParam().color = color;
}