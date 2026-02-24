#include "stdafx.h"
#include "DevicePhysicsConstant.h"

#include "../MCore_simulate/SimulateManager.h"

__constant__ DevicePhysicsData physicsData;

REGISTRY_DEVICE_CONSTANT(DevicePhysicsConstant)

DevicePhysicsConstant::DevicePhysicsConstant(ISimulateManager* pSimulateManager)
	: simulate::DeviceConstant<PhysicsData>{ pSimulateManager }
{}

void DevicePhysicsConstant::Initialize()
{
	DevicePhysicsData hostData;
	hostData.gravity = { 0.0, 0.0, -9.81 };
	hostData.dt = 0.01;
	CUDA_CHECK(cudaMemcpyToSymbol(physicsData, &hostData, sizeof(DevicePhysicsData)));
}

void DevicePhysicsConstant::OnModifyDB(const DB* prevDB, const DB* newDB)
{
	DevicePhysicsData hostData;
	hostData.gravity = newDB->gravity;
	hostData.dt = newDB->dt;
	CUDA_CHECK(cudaMemcpyToSymbol(physicsData, &hostData, sizeof(DevicePhysicsData)));
}
