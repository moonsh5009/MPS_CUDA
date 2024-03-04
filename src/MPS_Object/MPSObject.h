#pragma once

#include "../MCUDA_Lib/MCUDAGLBuffer.h"

#include "HeaderPre.h"

namespace mps
{
	struct ObjectParam
	{
		size_t size;
	};

	class ObjectResource
	{
	public:
		ObjectResource() = delete;
		ObjectResource(const size_t size) : m_size{ size } {}
		~ObjectResource() = default;
		ObjectResource(const ObjectResource&) = delete;
		ObjectResource(ObjectResource&&) = default;
		ObjectResource& operator=(const ObjectResource&) = delete;
		ObjectResource& operator=(ObjectResource&&) = default;

		virtual std::shared_ptr<ObjectParam> GetParam()
		{
			std::shared_ptr<ObjectParam> pParam = std::make_shared<ObjectParam>();
			SetParam(pParam);
			return pParam;
		}

	protected:
		void SetParam(std::shared_ptr<ObjectParam> pParam)
		{
			pParam->size = m_size;
		}

	private:
		size_t m_size;
	};

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

		virtual std::shared_ptr<ObjectResource> GetObjectResource();

	public:
		size_t m_size;
	};
}

#include "HeaderPost.h"