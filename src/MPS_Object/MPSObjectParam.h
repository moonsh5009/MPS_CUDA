#pragma once

#include "MPSDef.h"
#include "../MCUDA_Lib/MCUDAGLBuffer.h"

namespace mps
{
	struct ObjectParam
	{
		size_t size;
		size_t phaseID;

		glm::fvec4* pColor;
		REAL3* pPosition;
		REAL* pMass;
		REAL3* pVelocity;
		REAL3* pForce;
	};

	class ObjectResource
	{
	public:
		ObjectResource() = delete;
		ObjectResource(const size_t size, const size_t phaseID, mcuda::gl::DeviceResource<glm::fvec4>&& color, mcuda::gl::DeviceResource<REAL3>&& pos,
			REAL* mass, REAL3* velocity, REAL3* force) :
			m_size{ size }, m_phaseID{ phaseID }, m_color{ std::move(color) }, m_position{ std::move(pos) }, m_mass{ mass }, m_velocity{ velocity }, m_force{ force }
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

			m_pParam->size = m_size;
			m_pParam->phaseID = m_phaseID;
			m_pParam->pColor = m_color.GetData();
			m_pParam->pPosition = m_position.GetData();
			m_pParam->pMass = m_mass;
			m_pParam->pVelocity = m_velocity;
			m_pParam->pForce = m_force;
		}

	protected:
		std::shared_ptr<ObjectParam> m_pParam;
		size_t m_size;
		size_t m_phaseID;
		
		mcuda::gl::DeviceResource<glm::fvec4> m_color;
		mcuda::gl::DeviceResource<REAL3> m_position;
		REAL* m_mass;
		REAL3* m_velocity;
		REAL3* m_force;
	};
}