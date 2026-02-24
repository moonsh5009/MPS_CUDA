#pragma once

#include "../MPS_database/ClothDynamics.h"

#include "DeviceDynamicsSystemContainer.h"

#include "HeaderPre.h"

struct BendEdge
{
	IndexType v0, v1;
	IndexType v2, v3;
};

struct DeviceClothDynamicsData
{
	simulate::DeviceObjectData info;
	double* stretchStiffness;
	double* bendStiffness;
	double* dampingCoeff;

	double* restLengths;
	mcuda::DeviceMultiArrayInfo restLengthInfo;

	BendEdge* bendEdges;
	mcuda::DeviceMultiArrayInfo bendEdgeInfo;
};

class __MY_EXT_CLASS__ DeviceClothDynamicsContainer : public simulate::DeviceContainer<ClothDynamics>
{
public:
	using DeviceData = DeviceClothDynamicsData;

	DeviceClothDynamicsContainer(ISimulateManager* pSimulateManager);

	void OnAddDB(const DB* newDB) override;
	void OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB) override;
	void OnDeleteDB(IndexType index, const DB* prevDB) override;

	DeviceData GetDeviceData() const
	{
		DeviceData data;
		data.info = simulate::DeviceContainer<ClothDynamics>::GetDeviceData();
		data.stretchStiffness = stretchStiffnessArray.GetBuffer().GetData();
		data.bendStiffness = bendStiffnessArray.GetBuffer().GetData();
		data.dampingCoeff = dampingCoeffArray.GetBuffer().GetData();

		data.restLengths = restLengths.GetBuffer().GetData();
		data.restLengthInfo = restLengths.GetDeviceInfo();

		data.bendEdges = bendEdges.GetBuffer().GetData();
		data.bendEdgeInfo = bendEdges.GetDeviceInfo();
		return data;
	}

	mcuda::DeviceSingleArray<IndexType> dynamicsSystemIndices;
	mcuda::HostSingleArray<IndexType> hostDynamicsSystemIndices;

	mcuda::DeviceSingleArray<double> stretchStiffnessArray;
	mcuda::DeviceSingleArray<double> bendStiffnessArray;
	mcuda::DeviceSingleArray<double> dampingCoeffArray;

	mcuda::HostSingleArray<double> hostStretchStiffnessArray;
	mcuda::HostSingleArray<double> hostBendStiffnessArray;
	mcuda::HostSingleArray<double> hostDampingCoeffArray;

	mcuda::DeviceMultiArray<double> restLengths;
	mcuda::DeviceMultiArray<BendEdge> bendEdges;

	size_t activeNodeCount = 0;
	size_t activeEdgeCount = 0;
	size_t activeBendEdgeCount = 0;

	simulate::DeviceReference<DeviceDynamicsSystemContainer> dynamicsSystem;
};

#include "HeaderPost.h"
