#include "stdafx.h"
#include "MPSSPHMaterial.h"

namespace
{
	constexpr auto uNumParticlesByteLength = sizeof(size_t);
	constexpr auto uRadiusByteLength = sizeof(float);
	constexpr auto uMassByteLength = sizeof(float);
	constexpr auto uDensityByteLength = sizeof(float);
	constexpr auto uColorByteLength = sizeof(glm::fvec4);

	constexpr auto uNumParticlesOffset = 0ull;
	constexpr auto uRadiusOffset = uNumParticlesOffset + uNumParticlesByteLength;
	constexpr auto uMassOffset = uRadiusOffset + uRadiusByteLength;
	constexpr auto uDensityOffset = uMassOffset + uMassByteLength;
	constexpr auto uColorOffset = uDensityOffset + uDensityByteLength;
}

mps::SPHMaterial::SPHMaterial() : mps::VirtualMaterial<mps::SPHMaterialParam>{}
{
	SetParam({ 0ull, 0.1f, 0.0f, 0.0f, glm::fvec4{ 0.3f, 0.8f, 0.2f, 1.0f } });
}

void mps::SPHMaterial::SetParticleSize(const uint32_t size)
{
	GetHost().numParticles = size;
	GetDevice().CopyFromHost(GetHost(), uNumParticlesOffset, uNumParticlesOffset, uNumParticlesByteLength);
}

void mps::SPHMaterial::SetRadius(const float radius)
{
	GetHost().radius = radius;
	GetDevice().CopyFromHost(GetHost(), uRadiusOffset, uRadiusOffset, uRadiusByteLength);
}

void mps::SPHMaterial::SetMass(const float mass)
{
	GetHost().mass = mass;
	GetDevice().CopyFromHost(GetHost(), uMassOffset, uMassOffset, uMassByteLength);
}

void mps::SPHMaterial::SetDensity(const float density)
{
	GetHost().density = density;
	GetDevice().CopyFromHost(GetHost(), uDensityOffset, uDensityOffset, uDensityByteLength);
}

void mps::SPHMaterial::SetColor(const glm::fvec4& color)
{
	GetHost().color = color;
	GetDevice().CopyFromHost(GetHost(), uColorOffset, uColorOffset, uColorByteLength);
}