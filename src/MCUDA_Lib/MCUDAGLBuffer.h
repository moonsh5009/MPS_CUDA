#pragma once

#include "MGLBuffer.h"

#include "MCUDAHelper.h"
#include <cuda_gl_interop.h>

namespace mcuda
{
	namespace gl
	{
		struct CudaGLResource
		{
			bool bMapping = false;
			cudaGraphicsResource* resource = nullptr;
		};

		template<typename T>
		class DeviceResource
		{
		public:
			DeviceResource() = delete;
			DeviceResource(T* ptr, size_t size, std::shared_ptr<CudaGLResource> pCuRes) :
				m_ptr{ ptr }, m_size{ size }, m_pCuRes{ pCuRes }
			{
			}
			~DeviceResource()
			{
				const auto pCuRes = m_pCuRes.lock();
				if (!pCuRes) return;

				if (pCuRes->bMapping == false) { assert(false); return; }
				if (cudaGraphicsUnmapResources(1, &pCuRes->resource, 0) != cudaSuccess) { assert(false); return; }
				pCuRes->bMapping = false;
			}
			DeviceResource(const DeviceResource&) = delete;
			DeviceResource& operator=(const DeviceResource&) = delete;
			DeviceResource(DeviceResource&& other) = default;
			DeviceResource& operator=(DeviceResource&& other) = default;

		public:
			T* GetData() const { return m_ptr; }
			size_t GetSize() const { return m_size; }

		private:
			std::weak_ptr<CudaGLResource> m_pCuRes;
			T* m_ptr;
			size_t m_size;
		};

		template<typename T, GLenum TYPE = GL_DYNAMIC_DRAW>
		class Buffer : public mgl::Buffer<T, TYPE>
		{
		public:
			Buffer() : mgl::Buffer<T, TYPE>{}, m_pCuRes{ std::make_shared<CudaGLResource>() }
			{
			}
			Buffer(const size_t size) : mgl::Buffer<T, TYPE>{ size }, m_pCuRes{ std::make_shared<CudaGLResource>() }
			{
				CudaRegister();
			}
			Buffer(const std::vector<T>& host) : mgl::Buffer<T, TYPE>{ host }, m_pCuRes{ std::make_shared<CudaGLResource>() }
			{
				CudaRegister();
			}
			Buffer(const Buffer&) = delete;
			Buffer(Buffer&& other)
			{
				*this = std::move(other);
			}
			Buffer& operator=(const Buffer&) = delete;
			Buffer& operator=(Buffer&& other)
			{
				if (this != &other)
				{
					this->m_id = other.m_id;
					this->m_size = other.m_size;
					this->m_capacity = other.m_capacity;
					other.m_id = 0u;
					other.m_size = 0ull;
					other.m_capacity = 0ull;

					this->m_pCuRes = std::move(other.m_pCuRes);
				}
				return *this;
			}
			virtual ~Buffer()
			{
				Destroy();
			}

		private:
			void CudaRegister();
			void CudaUnRegister();

		public:
			virtual void Destroy() override;
			virtual void Resize(const size_t size) override;
			virtual void Resize(const std::vector<T>& host) override;

			std::optional<DeviceResource<T>> GetDeviceResource() const;

		private:
			std::shared_ptr<CudaGLResource> m_pCuRes;
		};
	}
}

#include "MCUDAGLBuffer.inl"