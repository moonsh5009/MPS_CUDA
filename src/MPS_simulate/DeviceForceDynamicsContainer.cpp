#include "stdafx.h"
#include "DeviceForceDynamicsContainer.h"

REGISTRY_DEVICE_CONTAINER(DeviceForceDynamicsContainer)

DeviceForceDynamicsContainer::DeviceForceDynamicsContainer(ISimulateManager* pSimulateManager)
	: simulate::DeviceContainer<ForceDynamics>{ pSimulateManager }
	, activeCount{ 0 }
	, mesh{ pSimulateManager }
	, kinetic{ pSimulateManager }
{}

void DeviceForceDynamicsContainer::OnAddDB(const DB* newDB)
{
	const auto meshIndex = mesh->GetIndex(newDB->mesh.GetKey());
	const auto kineticIndex = kinetic->GetIndex(newDB->kinetic.GetKey());

	meshIndices.Insert(meshIndex);
	kineticIndices.Insert(kineticIndex);

	hostMeshIndices.Insert(meshIndex);
	hostKineticIndices.Insert(kineticIndex);

	activeCount += newDB->mesh->nodes.size();
}

void DeviceForceDynamicsContainer::OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB)
{
	const auto meshIndex = mesh->GetIndex(newDB->mesh.GetKey());
	const auto kineticIndex = kinetic->GetIndex(newDB->kinetic.GetKey());

	meshIndices.Set(index, meshIndex);
	kineticIndices.Set(index, kineticIndex);

	hostMeshIndices.Set(index, meshIndex);
	hostKineticIndices.Set(index, kineticIndex);

	activeCount -= prevDB->mesh->nodes.size();
	activeCount += newDB->mesh->nodes.size();
}

void DeviceForceDynamicsContainer::OnDeleteDB(IndexType index, const DB* prevDB)
{
	meshIndices.Remove(index);
	kineticIndices.Remove(index);

	hostMeshIndices.Remove(index);
	hostKineticIndices.Remove(index);

	activeCount -= prevDB->mesh->nodes.size();
}
