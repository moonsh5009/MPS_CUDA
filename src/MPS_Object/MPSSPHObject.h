#pragma once

#include "MPSParticleObject.h"
#include <thrust/device_vector.h>

#include "HeaderPre.h"

namespace mps
{
	/*struct SPHParam : public ParticleParam
	{
		thrust::device_vector<float>::iterator density;
		thrust::device_vector<float>::iterator pressure;
		thrust::device_vector<float>::iterator factor;
	};

	struct SPHResource : public ParticleResource
	{
	public:
		SPHResource() = delete;
		SPHResource(std::shared_ptr<ParticleResource> pSuper,
			thrust::device_vector<float>::iterator density, thrust::device_vector<float>::iterator pressure, thrust::device_vector<float>::iterator factor) :
			ParticleResource{ std::move(*pSuper) },
			m_density{ density }, m_pressure{ pressure }, m_factor{ factor }
		{
		}
		~SPHResource() = default;
		SPHResource(const SPHResource&) = delete;
		SPHResource(SPHResource&&) = default;
		SPHResource& operator=(const SPHResource&) = delete;
		SPHResource& operator=(SPHResource&&) = default;

		virtual std::shared_ptr<ObjectParam> GetParam() override
		{
			std::shared_ptr<SPHParam> pParam = std::make_shared<SPHParam>();
			SetParam(pParam);
			return pParam;
		}

	protected:
		void SetParam(std::shared_ptr<SPHParam> pParam)
		{
			ParticleResource::SetParam(std::static_pointer_cast<ParticleParam>(pParam));

			pParam->density = m_density;
			pParam->pressure = m_pressure;
			pParam->factor = m_factor;
		}

	private:
		thrust::device_vector<float>::iterator m_density;
		thrust::device_vector<float>::iterator m_pressure;
		thrust::device_vector<float>::iterator m_factor;
	};*/

	struct SPHParam : public ParticleParam
	{
		float* mass;
		float* density;
		float* pressure;
		float* factor;
	};

	struct SPHResource : public ParticleResource
	{
	public:
		SPHResource() = delete;
		SPHResource(std::shared_ptr<ParticleResource> pSuper, float* mass, float* density, float* pressure, float* factor) :
			ParticleResource{ std::move(*pSuper) },
			m_mass{ mass }, m_density{ density }, m_pressure{ pressure }, m_factor{ factor }
		{
		}
		~SPHResource() = default;
		SPHResource(const SPHResource&) = delete;
		SPHResource(SPHResource&&) = default;
		SPHResource& operator=(const SPHResource&) = delete;
		SPHResource& operator=(SPHResource&&) = default;

		virtual std::shared_ptr<ObjectParam> GetParam() override
		{
			std::shared_ptr<SPHParam> pParam = std::make_shared<SPHParam>();
			SetParam(pParam);
			return pParam;
		}

	protected:
		void SetParam(std::shared_ptr<SPHParam> pParam)
		{
			ParticleResource::SetParam(std::static_pointer_cast<ParticleParam>(pParam));

			pParam->mass = m_mass;
			pParam->density = m_density;
			pParam->pressure = m_pressure;
			pParam->factor = m_factor;
		}

	private:
		float* m_mass;
		float* m_density;
		float* m_pressure;
		float* m_factor;
	};

	class __MY_EXT_CLASS__ SPHObject : public ParticleObject
	{
	public:
		SPHObject();
		~SPHObject() = default;
		SPHObject(const SPHObject&) = delete;
		SPHObject(SPHObject&&) = default;
		SPHObject& operator=(const SPHObject&) = delete;
		SPHObject& operator=(SPHObject&&) = default;

	public:
		virtual void Clear() override;
		virtual void Resize(const size_t size) override;

		virtual std::shared_ptr<ObjectResource> GetObjectResource() override;

	public:
		thrust::device_vector<float> m_mass;
		thrust::device_vector<float> m_density;
		thrust::device_vector<float> m_pressure;
		thrust::device_vector<float> m_factor;
	};
}

#include "HeaderPost.h"