#include "stdafx.h"
#include "MPSMeshMaterial.h"

mps::MeshMaterial::MeshMaterial() : mps::VirtualMaterial<mps::MeshMaterialParam>{}
{
	SetParam(0.1, 1.0);
	GetParam().viscosity = 0.0;
	GetParam().surfaceTension = 0.0;
	GetParam().frontColor = { 0.0f, 0.0f, 0.0f, 1.0f };
	GetParam().backColor = { 0.0f, 0.0f, 0.0f, 1.0f };
}

void mps::MeshMaterial::SetParam(const REAL radius, const REAL density)
{
	GetParam().radius = radius;
	const auto volume = radius * radius * radius / 48.0 * 3.141592;
	GetParam().density = density;
	GetParam().mass = density * volume;
}