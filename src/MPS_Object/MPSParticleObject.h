#pragma once

#include "MPSObject.h"

#include "HeaderPre.h"

namespace mps
{
	struct ParticleParam : public ObjectParam
	{
		glm::dvec3* pos;
		float* radius;
		glm::fvec4* color;
	};

	struct ParticleResource : public ObjectResource
	{
	public:
		ParticleResource() = delete;
		ParticleResource(std::shared_ptr<ObjectResource> pSuper,
			mcuda::gl::DeviceResource<glm::dvec3>&& pos, mcuda::gl::DeviceResource<float>&& radius, mcuda::gl::DeviceResource<glm::fvec4>&& color) :
			ObjectResource{ std::move(*pSuper) }, m_pos{ std::move(pos) }, m_radius{ std::move(radius) }, m_color{ std::move(color) }
		{
		}
		~ParticleResource() = default;
		ParticleResource(const ParticleResource&) = delete;
		ParticleResource(ParticleResource&&) = default;
		ParticleResource& operator=(const ParticleResource&) = delete;
		ParticleResource& operator=(ParticleResource&&) = default;

		virtual std::shared_ptr<ObjectParam> GetParam() override
		{
			std::shared_ptr<ParticleParam> pParam = std::make_shared<ParticleParam>();
			SetParam(pParam);
			return pParam;
		}

	protected:
		void SetParam(std::shared_ptr<ParticleParam> pParam)
		{
			ObjectResource::SetParam(std::static_pointer_cast<ObjectParam>(pParam));
			
			pParam->pos = m_pos.ptr;
			pParam->radius = m_radius.ptr;
			pParam->color = m_color.ptr;
		}

	private:
		mcuda::gl::DeviceResource<glm::dvec3> m_pos;
		mcuda::gl::DeviceResource<float> m_radius;
		mcuda::gl::DeviceResource<glm::fvec4> m_color;
	};

	class __MY_EXT_CLASS__ ParticleObject : public Object
	{
	public:
		ParticleObject();
		~ParticleObject() = default;
		ParticleObject(const ParticleObject&) = delete;
		ParticleObject(ParticleObject&&) = default;
		ParticleObject& operator=(const ParticleObject&) = delete;
		ParticleObject& operator=(ParticleObject&&) = default;

	public:
		virtual void Clear() override;
		virtual void Resize(const size_t size) override;

		virtual std::shared_ptr<ObjectResource> GetObjectResource() override;

	public:
		mcuda::gl::Buffer<glm::dvec3> m_pos;
		mcuda::gl::Buffer<float> m_radius;
		mcuda::gl::Buffer<glm::fvec4> m_color;
	};
}

#include "HeaderPost.h"