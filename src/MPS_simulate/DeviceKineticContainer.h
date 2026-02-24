#pragma once

#include "../MCore_simulate/DeviceContainer.h"

#include "../MPS_database/KineticData.h"

#include "HeaderPre.h"

struct DeviceKineticData
{
	simulate::DeviceObjectData info;
	mcuda::DeviceMultiArrayInfo arrayInfo;

	double* masses;
	double* invMasses;
	mcore::Vector3* velocities;
	mcore::Vector3* forces;

	IndexType* fixeds;
};

class __MY_EXT_CLASS__ DeviceKineticContainer : public simulate::DeviceContainer<KineticData>
{
public:
	using DeviceData = DeviceKineticData;

	DeviceKineticContainer(ISimulateManager* pSimulateManager);

	void OnAddDB(const DB* newDB) override;
	void OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB) override;
	void OnDeleteDB(IndexType index, const DB* prevDB) override;

	DeviceData GetDeviceData() const
	{
		DeviceData data;
		data.info = simulate::DeviceContainer<KineticData>::GetDeviceData();
		data.arrayInfo = buffers.GetDeviceInfo();
		data.masses = GetMasses().GetData();
		data.invMasses = GetInvMasses().GetData();
		data.velocities = GetVelocities().GetData();
		data.forces = GetForces().GetData();
		data.fixeds = GetFixeds().GetData();
		return data;
	}

	mcuda::DeviceBuffer<double>& GetMasses() { return buffers.GetBuffer<0>(); }
	mcuda::DeviceBuffer<double>& GetInvMasses() { return buffers.GetBuffer<1>(); }
	mcuda::DeviceBuffer<mcore::Vector3>& GetVelocities() { return buffers.GetBuffer<2>(); }
	mcuda::DeviceBuffer<mcore::Vector3>& GetForces() { return buffers.GetBuffer<3>(); }
	mcuda::DeviceBuffer<IndexType>& GetFixeds() { return fixeds.GetBuffer(); }

	const mcuda::DeviceBuffer<double>& GetMasses() const { return buffers.GetBuffer<0>(); }
	const mcuda::DeviceBuffer<double>& GetInvMasses() const { return buffers.GetBuffer<1>(); }
	const mcuda::DeviceBuffer<mcore::Vector3>& GetVelocities() const { return buffers.GetBuffer<2>(); }
	const mcuda::DeviceBuffer<mcore::Vector3>& GetForces() const { return buffers.GetBuffer<3>(); }
	const mcuda::DeviceBuffer<IndexType>& GetFixeds() const { return fixeds.GetBuffer(); }

	mcuda::DeviceMultiArray<
		double,
		double,
		mcore::Vector3,
		mcore::Vector3> buffers;

	mcuda::DeviceMultiArray<IndexType> fixeds;
};

#include "HeaderPost.h"
