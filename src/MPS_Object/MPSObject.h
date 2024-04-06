#pragma once

#include <thrust/device_vector.h>
#include "MPSObjectParam.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ Object
	{
	public:
		Object();
		~Object() = default;
		Object(const Object&) = delete;
		Object(Object&&) = default;
		Object& operator=(const Object&) = delete;
		Object& operator=(Object&&) = default;

	public:
		constexpr size_t GetSize() const { return m_size; };

		virtual void Clear();
		virtual void Resize(const size_t size);
		
		template<class T>
		std::shared_ptr<const T> GetDeviceResource()
		{
			std::shared_ptr<T> pResource = std::static_pointer_cast<T>(GenerateDeviceResource());
			if (!pResource) return nullptr;
			return std::const_pointer_cast<const T>(pResource);
		}

	protected:
		virtual std::shared_ptr<ObjectResource> GenerateDeviceResource();

	public:
		size_t m_size;

		mcuda::gl::Buffer<glm::fvec4> m_color;
		mcuda::gl::Buffer<REAL3> m_pos;
		thrust::device_vector<REAL> m_mass;
		thrust::device_vector<REAL3> m_velocity;
		thrust::device_vector<REAL3> m_force;
	};
}

#include "HeaderPost.h"