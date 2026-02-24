#pragma once

#include "../MCore_simulate/DeviceContainer.h"

#include "../MPS_database/ForceDynamics.h"

#include "DeviceMeshContainer.h"
#include "DeviceKineticContainer.h"

#include "HeaderPre.h"

struct DeviceForceDynamicsData
{
	simulate::DeviceObjectData info;
	DeviceMeshData meshInfo;
	DeviceKineticData kineticInfo;

	size_t count;
	IndexType* meshIndices;
	IndexType* kineticIndices;
};

class __MY_EXT_CLASS__ DeviceForceDynamicsContainer : public simulate::DeviceContainer<ForceDynamics>
{
public:
	using DeviceData = DeviceForceDynamicsData;

	DeviceForceDynamicsContainer(ISimulateManager* pSimulateManager);

	void OnAddDB(const DB* newDB) override;
	void OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB) override;
	void OnDeleteDB(IndexType index, const DB* prevDB) override;

	DeviceData GetDeviceData() const
	{
		DeviceData data;
		data.info = simulate::DeviceContainer<ForceDynamics>::GetDeviceData();
		data.meshInfo = mesh->GetDeviceData();
		data.kineticInfo = kinetic->GetDeviceData();

		data.count = GetMeshIndices().GetSize();
		data.meshIndices = GetMeshIndices().GetData();
		data.kineticIndices = GetKineticIndices().GetData();
		return data;
	}

	mcuda::DeviceBuffer<IndexType>& GetMeshIndices() { return meshIndices.GetBuffer(); }
	mcuda::DeviceBuffer<IndexType>& GetKineticIndices() { return kineticIndices.GetBuffer(); }
	std::vector<IndexType>& GetHostMeshIndices() { return hostMeshIndices.GetBuffer(); }
	std::vector<IndexType>& GetHostKineticIndices() { return hostKineticIndices.GetBuffer(); }

	const mcuda::DeviceBuffer<IndexType>& GetMeshIndices() const { return meshIndices.GetBuffer(); }
	const mcuda::DeviceBuffer<IndexType>& GetKineticIndices() const { return kineticIndices.GetBuffer(); }
	const std::vector<IndexType>& GetHostMeshIndices() const { return hostMeshIndices.GetBuffer(); }
	const std::vector<IndexType>& GetHostKineticIndices() const { return hostKineticIndices.GetBuffer(); }

	size_t activeCount;
	mcuda::DeviceSingleArray<IndexType> meshIndices;
	mcuda::DeviceSingleArray<IndexType> kineticIndices;
	mcuda::HostSingleArray<IndexType> hostMeshIndices;
	mcuda::HostSingleArray<IndexType> hostKineticIndices;

	simulate::DeviceReference<DeviceMeshContainer> mesh;
	simulate::DeviceReference<DeviceKineticContainer> kinetic;
};

#include "HeaderPost.h"
