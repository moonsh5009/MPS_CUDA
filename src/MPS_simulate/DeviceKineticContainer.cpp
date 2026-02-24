#include "stdafx.h"
#include "DeviceKineticContainer.h"

REGISTRY_DEVICE_CONTAINER(DeviceKineticContainer)

DeviceKineticContainer::DeviceKineticContainer(ISimulateManager* pSimulateManager)
	: simulate::DeviceContainer<KineticData>{ pSimulateManager }
{}

void DeviceKineticContainer::OnAddDB(const DB* newDB)
{
	buffers.Insert(
		newDB->masses,
		newDB->invMasses,
		newDB->velocities,
		newDB->forces);
	fixeds.Insert(
		newDB->fixeds);
}

void DeviceKineticContainer::OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB)
{
	buffers.Set(
		index,
		newDB->masses,
		newDB->invMasses,
		newDB->velocities,
		newDB->forces);
	fixeds.Set(
		index,
		newDB->fixeds);
}

void DeviceKineticContainer::OnDeleteDB(IndexType index, const DB* prevDB)
{
	buffers.Remove(index);
	fixeds.Remove(index);
}
