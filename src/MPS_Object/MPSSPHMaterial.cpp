#include "stdafx.h"
#include "MPSSPHMaterial.h"

mps::SPHMaterial::SPHMaterial() : mps::VirtualMaterial<mps::SPHMaterialParam>{}
{
	SetParam(0.1, 1.0);
	SetViscosity(0.05);
	SetSurfaceTension(70000.0);
	SetColor({ 0.3f, 0.8f, 0.2f, 1.0f });
}

void mps::SPHMaterial::SetParam(const REAL radius, const REAL density)
{
	GetParam().radius = radius;
	GetParam().volume = radius * radius * radius / 48.0 * 3.141592;
	GetParam().density = density;
	GetParam().mass = density * GetParam().volume;
}

void mps::SPHMaterial::SetRadius(const REAL radius)
{
	SetParam(radius, GetDensity());
}

void mps::SPHMaterial::SetViscosity(const REAL viscosity)
{
	GetParam().viscosity = viscosity;
}

void mps::SPHMaterial::SetSurfaceTension(const REAL surfaceTension)
{
	GetParam().surfaceTension = surfaceTension;
}

void mps::SPHMaterial::SetColor(const glm::fvec4& color)
{
	GetParam().color = color;
}