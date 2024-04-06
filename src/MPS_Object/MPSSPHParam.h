#pragma once

#include "MPSParticleParam.h"

namespace mps
{
	class SPHParam : public ParticleParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC SPHParam() : ParticleParam{}, m_pDensity{ nullptr }, m_pPressure{ nullptr }, m_pFactor{ nullptr } {}
		MCUDA_HOST_DEVICE_FUNC ~SPHParam() {};

	public:
		MCUDA_DEVICE_FUNC REAL& Density(uint32_t idx) { return m_pDensity[idx]; }
		MCUDA_DEVICE_FUNC REAL& Pressure(uint32_t idx) { return m_pPressure[idx]; }
		MCUDA_DEVICE_FUNC REAL& Factor(uint32_t idx) { return m_pFactor[idx]; }

		MCUDA_DEVICE_FUNC const REAL& Density(uint32_t idx) const { return m_pDensity[idx]; }
		MCUDA_DEVICE_FUNC const REAL& Pressure(uint32_t idx) const { return m_pPressure[idx]; }
		MCUDA_DEVICE_FUNC const REAL& Factor(uint32_t idx) const { return m_pFactor[idx]; }

	public:
		MCUDA_HOST_FUNC REAL* GetDensityArray() const { return m_pDensity; }
		MCUDA_HOST_FUNC REAL* GetPressureArray() const { return m_pPressure; }
		MCUDA_HOST_FUNC REAL* GetFactorArray() const { return m_pFactor; }

		MCUDA_HOST_FUNC void SetDensityArray(REAL* pDensity) { m_pDensity = pDensity; }
		MCUDA_HOST_FUNC void SetPressureArray(REAL* pPressure) { m_pPressure = pPressure; }
		MCUDA_HOST_FUNC void SetFactorArray(REAL* pFactor) { m_pFactor = pFactor; }

	private:
		REAL* m_pDensity;
		REAL* m_pPressure;
		REAL* m_pFactor;
	};

	struct SPHResource : public ParticleResource
	{
	public:
		SPHResource() = delete;
		SPHResource(std::shared_ptr<ParticleResource> pSuper, REAL* density, REAL* pressure, REAL* factor) :
			ParticleResource{ std::move(*pSuper) }, m_density{ density }, m_pressure{ pressure }, m_factor{ factor }
		{}
		~SPHResource() = default;
		SPHResource(const SPHResource&) = delete;
		SPHResource(SPHResource&&) = default;
		SPHResource& operator=(const SPHResource&) = delete;
		SPHResource& operator=(SPHResource&&) = default;

	public:
		std::weak_ptr<ParticleParam> GetSPHParam() const
		{
			return std::static_pointer_cast<SPHParam>(m_pParam);
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<SPHParam>();

			auto pParam = std::static_pointer_cast<SPHParam>(m_pParam);
			pParam->SetDensityArray(m_density);
			pParam->SetPressureArray(m_pressure);
			pParam->SetFactorArray(m_factor);
			ParticleResource::SetParam();
		}

	private:
		REAL* m_density;
		REAL* m_pressure;
		REAL* m_factor;
	};
}