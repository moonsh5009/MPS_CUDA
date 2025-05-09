template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::CudaRegister()
{
	if (this->GetSize() <= 0ull) return;
	assert(glIsBuffer(this->GetID()));
	CudaUnRegister();

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_pCuRes->resource, this->GetID(), cudaGraphicsMapFlagsNone));
}

template<typename T, GLenum TYPE>
void mcuda::gl::Buffer<T, TYPE>::CudaUnRegister()
{
	if (!m_pCuRes->resource) return;
	assert(m_pCuRes->bMapping == false);

	CUDA_CHECK(cudaGraphicsUnregisterResource(m_pCuRes->resource));
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
std::optional<mcuda::gl::DeviceResource<T>> mcuda::gl::Buffer<T, TYPE>::GetDeviceResource() const
{
	if (m_pCuRes->bMapping) return {};

	if (const auto error = cudaGraphicsMapResources(1, &m_pCuRes->resource, 0); error != cudaSuccess)
	{
		CUDA_CHECK(error);
		assert(false);
		return {};
	}

	T* ptr;
	size_t capacityByteLength;
	if (const auto error = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &capacityByteLength, m_pCuRes->resource); error != cudaSuccess)
	{
		CUDA_CHECK(error);
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