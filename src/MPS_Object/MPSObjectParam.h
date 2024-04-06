#pragma once

#include "MPSDef.h"
#include "../MCUDA_Lib/MCUDAGLBuffer.h"

namespace mps
{
	class ObjectParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC ObjectParam() : m_size{ 0ull }, 
			m_pColor{ nullptr }, m_pPos{ nullptr }, m_pMass{ nullptr }, m_pVelocity{ nullptr }, m_pForce{ nullptr } {}
		MCUDA_HOST_DEVICE_FUNC ~ObjectParam() {}

	public:
		MCUDA_HOST_DEVICE_FUNC size_t GetSize() const { return m_size; }

		MCUDA_DEVICE_FUNC glm::fvec4& Color(uint32_t idx) const { return m_pColor[idx]; }
		MCUDA_DEVICE_FUNC REAL3& Position(uint32_t idx) const { return m_pPos[idx]; }
		MCUDA_DEVICE_FUNC REAL& Mass(uint32_t idx) const { return m_pMass[idx]; }
		MCUDA_DEVICE_FUNC REAL3& Velocity(uint32_t idx) const { return m_pVelocity[idx]; }
		MCUDA_DEVICE_FUNC REAL3& Force(uint32_t idx) const { return m_pForce[idx]; }

	public:
		MCUDA_HOST_FUNC void SetSize(size_t size) { m_size = size; }

		MCUDA_HOST_FUNC glm::fvec4* GetColorArray() const { return m_pColor; }
		MCUDA_HOST_FUNC REAL3* GetPosArray() const { return m_pPos; }
		MCUDA_HOST_FUNC REAL* GetMassArray() const { return m_pMass; }
		MCUDA_HOST_FUNC REAL3* GetVelocityArray() const { return m_pVelocity; }
		MCUDA_HOST_FUNC REAL3* GetForceArray() const { return m_pForce; }

		MCUDA_HOST_FUNC void SetColorArray(glm::fvec4* pColor) { m_pColor = pColor; }
		MCUDA_HOST_FUNC void SetPosArray(REAL3* pPos) { m_pPos = pPos; }
		MCUDA_HOST_FUNC void SetMassArray(REAL* pMass) { m_pMass = pMass; }
		MCUDA_HOST_FUNC void SetVelocityArray(REAL3* pVelocity) { m_pVelocity = pVelocity; }
		MCUDA_HOST_FUNC void SetForceArray(REAL3* pForce) { m_pForce = pForce; }

	private:
		size_t m_size;

		glm::fvec4* m_pColor;
		REAL3* m_pPos;
		REAL* m_pMass;
		REAL3* m_pVelocity;
		REAL3* m_pForce;
	};

	class ObjectResource
	{
	public:
		ObjectResource() = delete;
		ObjectResource(const size_t size, mcuda::gl::DeviceResource<glm::fvec4>&& color, mcuda::gl::DeviceResource<REAL3>&& pos,
			REAL* mass, REAL3* velocity, REAL3* force) :
			m_size{ size }, m_color{ std::move(color) }, m_pos{ std::move(pos) }, m_mass{ mass }, m_velocity{ velocity }, m_force{ force }
		{}
		~ObjectResource() = default;
		ObjectResource(const ObjectResource&) = delete;
		ObjectResource(ObjectResource&&) = default;
		ObjectResource& operator=(const ObjectResource&) = delete;
		ObjectResource& operator=(ObjectResource&&) = default;

	public:
		std::weak_ptr<ObjectParam> GetObjectParam() const
		{
			return m_pParam;
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<ObjectParam>();

			m_pParam->SetSize(m_size);
			m_pParam->SetColorArray(m_color.GetData());
			m_pParam->SetPosArray(m_pos.GetData());
			m_pParam->SetMassArray(m_mass);
			m_pParam->SetVelocityArray(m_velocity);
			m_pParam->SetForceArray(m_force);
		}

	protected:
		std::shared_ptr<ObjectParam> m_pParam;
		size_t m_size;

		mcuda::gl::DeviceResource<glm::fvec4> m_color;
		mcuda::gl::DeviceResource<REAL3> m_pos;
		REAL* m_mass;
		REAL3* m_velocity;
		REAL3* m_force;
	};
}