#pragma once

#include "../MPS_database/ClothData.h"

#include "DeviceMeshContainer.h"
#include "DeviceKineticContainer.h"

#include "HeaderPre.h"

struct BendEdge
{
	IndexType v0, v1;
	IndexType v2, v3;
};

struct DeviceClothData
{
	simulate::DeviceObjectData info;
	DeviceMeshData meshInfo;
	DeviceKineticData kineticInfo;
	IndexType* meshIndices;
	IndexType* kineticIndices;

	double* stretchStiffness;
	double* bendStiffness;
	double* dampingCoeff;

	double* restLengths;
	mcuda::DeviceMultiArrayInfo restLengthInfo;

	BendEdge* bendEdges;
	mcuda::DeviceMultiArrayInfo bendEdgeInfo;
};

class __MY_EXT_CLASS__ DeviceClothContainer : public simulate::DeviceContainer<ClothData>
{
public:
	using DeviceData = DeviceClothData;

	DeviceClothContainer(ISimulateManager* pSimulateManager);

	void OnAddDB(const DB* newDB) override;
	void OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB) override;
	void OnDeleteDB(IndexType index, const DB* prevDB) override;

	DeviceData GetDeviceData() const
	{
		DeviceData data;
		data.info = simulate::DeviceContainer<ClothData>::GetDeviceData();
		data.meshInfo = mesh->GetDeviceData();
		data.kineticInfo = kinetic->GetDeviceData();
		data.meshIndices = GetMeshIndices().GetData();
		data.kineticIndices = GetKineticIndices().GetData();

		data.stretchStiffness = stretchStiffnessArray.GetBuffer().GetData();
		data.bendStiffness = bendStiffnessArray.GetBuffer().GetData();
		data.dampingCoeff = dampingCoeffArray.GetBuffer().GetData();

		data.restLengths = restLengths.GetBuffer().GetData();
		data.restLengthInfo = restLengths.GetDeviceInfo();

		data.bendEdges = bendEdges.GetBuffer().GetData();
		data.bendEdgeInfo = bendEdges.GetDeviceInfo();
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

	mcuda::DeviceSingleArray<IndexType> meshIndices;
	mcuda::DeviceSingleArray<IndexType> kineticIndices;
	mcuda::HostSingleArray<IndexType> hostMeshIndices;
	mcuda::HostSingleArray<IndexType> hostKineticIndices;

	mcuda::DeviceSingleArray<double> stretchStiffnessArray;
	mcuda::DeviceSingleArray<double> bendStiffnessArray;
	mcuda::DeviceSingleArray<double> dampingCoeffArray;
	mcuda::HostSingleArray<IndexType> cgMaxIterationsArray;

	mcuda::DeviceMultiArray<double> restLengths;
	mcuda::DeviceMultiArray<BendEdge> bendEdges;

	size_t activeNodeCount = 0;
	size_t activeEdgeCount = 0;
	size_t activeBendEdgeCount = 0;

	simulate::DeviceReference<DeviceMeshContainer> mesh;
	simulate::DeviceReference<DeviceKineticContainer> kinetic;
};

#include "HeaderPost.h"
