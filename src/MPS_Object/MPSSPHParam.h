#pragma once

#include "MPSParticleParam.h"

namespace mps
{
	class SPHParam : public ParticleParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC SPHParam() : ParticleParam{},
			m_pDensity{ nullptr }, m_pPressure{ nullptr }, m_pFactorA{ nullptr },
			m_pSmallDensity{ nullptr }, m_pSmallPressure{ nullptr }, m_pSurfaceTensor{ nullptr }, m_pColorField{ nullptr }
		{}
		MCUDA_HOST_DEVICE_FUNC ~SPHParam() {};

	public:
		MCUDA_DEVICE_FUNC REAL& Density(uint32_t idx) { return m_pDensity[idx]; }
		MCUDA_DEVICE_FUNC REAL& Pressure(uint32_t idx) { return m_pPressure[idx]; }
		MCUDA_DEVICE_FUNC REAL& FactorA(uint32_t idx) { return m_pFactorA[idx]; }

		MCUDA_DEVICE_FUNC REAL& SmallDensity(uint32_t idx) { return m_pSmallDensity[idx]; }
		MCUDA_DEVICE_FUNC REAL& SmallPressure(uint32_t idx) { return m_pSmallPressure[idx]; }
		MCUDA_DEVICE_FUNC REAL3x3& SurfaceTensor(uint32_t idx) { return m_pSurfaceTensor[idx]; }
		MCUDA_DEVICE_FUNC REAL& ColorField(uint32_t idx) { return m_pColorField[idx]; }

		MCUDA_DEVICE_FUNC const REAL& Density(uint32_t idx) const { return m_pDensity[idx]; }
		MCUDA_DEVICE_FUNC const REAL& Pressure(uint32_t idx) const { return m_pPressure[idx]; }
		MCUDA_DEVICE_FUNC const REAL& FactorA(uint32_t idx) const { return m_pFactorA[idx]; }

		MCUDA_DEVICE_FUNC const REAL& SmallDensity(uint32_t idx) const { return m_pSmallDensity[idx]; }
		MCUDA_DEVICE_FUNC const REAL& SmallPressure(uint32_t idx) const { return m_pSmallPressure[idx]; }
		MCUDA_DEVICE_FUNC const REAL3x3& SurfaceTensor(uint32_t idx) const { return m_pSurfaceTensor[idx]; }
		MCUDA_DEVICE_FUNC const REAL& ColorField(uint32_t idx) const { return m_pColorField[idx]; }

	public:
		MCUDA_HOST_FUNC REAL* GetDensityArray() const { return m_pDensity; }
		MCUDA_HOST_FUNC REAL* GetPressureArray() const { return m_pPressure; }
		MCUDA_HOST_FUNC REAL* GetFactorAArray() const { return m_pFactorA; }

		MCUDA_HOST_FUNC REAL* GetSmallDensityArray() const { return m_pSmallDensity; }
		MCUDA_HOST_FUNC REAL* GetSmallPressureArray() const { return m_pSmallPressure; }
		MCUDA_HOST_FUNC REAL3x3* GetSurfaceTensorArray() const { return m_pSurfaceTensor; }
		MCUDA_HOST_FUNC REAL* GetColorFieldArray() const { return m_pColorField; }

		MCUDA_HOST_FUNC void SetDensityArray(REAL* pDensity) { m_pDensity = pDensity; }
		MCUDA_HOST_FUNC void SetPressureArray(REAL* pPressure) { m_pPressure = pPressure; }
		MCUDA_HOST_FUNC void SetFactorAArray(REAL* pFactorA) { m_pFactorA = pFactorA; }

		MCUDA_HOST_FUNC void SetSmallDensityArray(REAL* pSmallDensity) { m_pSmallDensity = pSmallDensity; }
		MCUDA_HOST_FUNC void SetSmallPressureArray(REAL* pSmallPressure) { m_pSmallPressure = pSmallPressure; }
		MCUDA_HOST_FUNC void SetSurfaceTensorArray(REAL3x3* pSurfaceTensor) { m_pSurfaceTensor = pSurfaceTensor; }
		MCUDA_HOST_FUNC void SetColorFieldArray(REAL* pColorField) { m_pColorField = pColorField; }

	private:
		REAL* MCUDA_RESTRICT m_pDensity;
		REAL* MCUDA_RESTRICT m_pPressure;
		REAL* MCUDA_RESTRICT m_pFactorA;

		REAL* MCUDA_RESTRICT m_pSmallDensity;
		REAL* MCUDA_RESTRICT m_pSmallPressure;
		REAL3x3* MCUDA_RESTRICT m_pSurfaceTensor;
		REAL* MCUDA_RESTRICT m_pColorField;
	};

	struct SPHResource : public ParticleResource
	{
	public:
		SPHResource() = delete;
		SPHResource(std::shared_ptr<ParticleResource> pSuper,
			REAL* density, REAL* pressure, REAL* factorA, REAL* smallDensity, REAL* smallPressure, REAL3x3* surfaceTensor, REAL* colorField) :
			ParticleResource{ std::move(*pSuper) }, m_density{ density }, m_pressure{ pressure }, m_factorA{ factorA },
			m_smallDensity{ density }, m_smallPressure{ pressure }, m_surfaceTensor{ surfaceTensor }, m_colorField{ colorField }
		{}
		~SPHResource() = default;
		SPHResource(const SPHResource&) = delete;
		SPHResource(SPHResource&&) = default;
		SPHResource& operator=(const SPHResource&) = delete;
		SPHResource& operator=(SPHResource&&) = default;

	public:
		std::weak_ptr<SPHParam> GetSPHParam() const
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
			pParam->SetFactorAArray(m_factorA);

			pParam->SetSmallDensityArray(m_smallDensity);
			pParam->SetSmallPressureArray(m_smallPressure);
			pParam->SetSurfaceTensorArray(m_surfaceTensor);
			pParam->SetColorFieldArray(m_colorField);
			ParticleResource::SetParam();
		}

	private:
		REAL* m_density;
		REAL* m_pressure;
		REAL* m_factorA;

		REAL* m_smallDensity;
		REAL* m_smallPressure;
		REAL3x3* m_surfaceTensor;
		REAL* m_colorField;
	};
}