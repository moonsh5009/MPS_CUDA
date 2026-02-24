#include "stdafx.h"
#include "DeviceDynamicsSystemContainer.h"

REGISTRY_DEVICE_CONTAINER(DeviceDynamicsSystemContainer)

DeviceDynamicsSystemContainer::DeviceDynamicsSystemContainer(ISimulateManager* pSimulateManager)
	: simulate::DeviceContainer<DynamicsSystem>{ pSimulateManager }
	, mesh{ pSimulateManager }
	, kinetic{ pSimulateManager }
{}

void DeviceDynamicsSystemContainer::OnAddDB(const DB* newDB)
{
	const auto meshIndex = mesh->GetIndex(newDB->mesh.GetKey());
	const auto kineticIndex = kinetic->GetIndex(newDB->kinetic.GetKey());

	meshIndices.Insert(meshIndex);
	kineticIndices.Insert(kineticIndex);
	hostMeshIndices.Insert(meshIndex);
	hostKineticIndices.Insert(kineticIndex);

	maxIterationsArray.Insert(newDB->maxIterations);
	toleranceArray.Insert(newDB->tolerance);

	const auto nodeCount = newDB->mesh->nodes.size();
	entryNodeOffsets.push_back(totalNodeCount);
	entryNodeCounts.push_back(nodeCount);
	totalNodeCount += nodeCount;
}

void DeviceDynamicsSystemContainer::OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB)
{
	const auto meshIndex = mesh->GetIndex(newDB->mesh.GetKey());
	const auto kineticIndex = kinetic->GetIndex(newDB->kinetic.GetKey());

	meshIndices.Set(index, meshIndex);
	kineticIndices.Set(index, kineticIndex);
	hostMeshIndices.Set(index, meshIndex);
	hostKineticIndices.Set(index, kineticIndex);

	maxIterationsArray.Set(index, newDB->maxIterations);
	toleranceArray.Set(index, newDB->tolerance);

	totalNodeCount -= entryNodeCounts[index];
	const auto nodeCount = newDB->mesh->nodes.size();
	entryNodeCounts[index] = nodeCount;
	totalNodeCount += nodeCount;

	size_t offset = 0;
	for (size_t i = 0; i < entryNodeCounts.size(); ++i)
	{
		entryNodeOffsets[i] = offset;
		offset += entryNodeCounts[i];
	}
}

void DeviceDynamicsSystemContainer::OnDeleteDB(IndexType index, const DB* prevDB)
{
	totalNodeCount -= entryNodeCounts[index];

	meshIndices.Remove(index);
	kineticIndices.Remove(index);
	hostMeshIndices.Remove(index);
	hostKineticIndices.Remove(index);

	maxIterationsArray.Remove(index);
	toleranceArray.Remove(index);

	entryNodeOffsets.erase(entryNodeOffsets.begin() + index);
	entryNodeCounts.erase(entryNodeCounts.begin() + index);

	size_t offset = 0;
	for (size_t i = 0; i < entryNodeCounts.size(); ++i)
	{
		entryNodeOffsets[i] = offset;
		offset += entryNodeCounts[i];
	}
}
