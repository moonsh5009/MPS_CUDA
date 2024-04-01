template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::CudaRegister()
{
	if (this->GetSize() <= 0ull) return;
	CudaUnRegister();

	const auto error = cudaGraphicsGLRegisterBuffer(&m_pCuRes->resource, this->GetID(), cudaGraphicsMapFlagsNone);
	assert(error == cudaSuccess);
}

template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::CudaUnRegister()
{
	if (!m_pCuRes->resource) return;
	assert(m_pCuRes->bMapping == false);

	const auto error = cudaGraphicsUnregisterResource(m_pCuRes->resource);
	assert(error == cudaSuccess);
	m_pCuRes->resource = nullptr;
}

template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::Destroy()
{
	assert(m_pCuRes->bMapping == false);
	CudaUnRegister();
	mgl::Buffer<T, TYPE>::Destroy();
}

template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::Resize(const size_t size)
{
	const auto curCapacity = this->GetCapacity();
	mgl::Buffer<T, TYPE>::Resize(size);

	if (this->GetCapacity() != curCapacity)
		CudaRegister();
}

template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::Resize(const std::vector<T>& host)
{
	const auto curCapacity = this->GetCapacity();
	mgl::Buffer<T, TYPE>::Resize(host);

	if (this->GetCapacity() != curCapacity)
		CudaRegister();
}

template<typename T, GLenum TYPE>
std::optional<mcuda::gl::DeviceResource<T>> mcuda::gl::Buffer<T, TYPE>::GetDeviceResource()
{
	if (m_pCuRes->bMapping) return {};

	if (cudaGraphicsMapResources(1, &m_pCuRes->resource, 0) != cudaSuccess)
	{
		assert(false);
		return {};
	}

	T* ptr;
	size_t capacityByteLength;
	if (cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &capacityByteLength, m_pCuRes->resource) != cudaSuccess)
	{
		assert(false);
		return {};
	}

	if (capacityByteLength != this->GetCapacityByteLength())
	{
		assert(false);
		cudaGraphicsUnmapResources(1, &m_pCuRes->resource, 0);
		return {};
	}
	m_pCuRes->bMapping = true;

	return DeviceResource<T>(ptr, this->GetSize(), m_pCuRes);
}