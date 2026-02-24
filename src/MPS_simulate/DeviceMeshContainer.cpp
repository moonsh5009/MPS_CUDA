#include "stdafx.h"
#include "DeviceMeshContainer.h"

#include <ranges>

REGISTRY_DEVICE_CONTAINER(DeviceMeshContainer)

DeviceMeshContainer::DeviceMeshContainer(ISimulateManager* pSimulateManager)
	: simulate::DeviceContainer<MeshData>{ pSimulateManager }
	, faceIndices{ vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, edgeIndices{ vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, nodeIndices{ vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, nodeBuffers{ vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, faceDrawIndirects{ vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, edgeDrawIndirects{ vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, nodeDrawIndirects{ vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer }
	, bvhTree{ std::make_unique<simulate::TriangleBVHTree>() }
{}

void DeviceMeshContainer::OnAddDB(const DB* newDB)
{
	faceIndices.Insert(newDB->faceIndices);
	edgeIndices.Insert(newDB->edgeIndices);

	const auto range = std::views::iota(
		static_cast<IndexType>(0),
		static_cast<IndexType>(newDB->nodes.size()));
	std::vector<IndexType> indices(range.begin(), range.end());
	nodeIndices.Insert(indices);

	nodeBuffers.Insert(
		newDB->nodes,
		newDB->normals,
		newDB->triangleAttributes,
		newDB->lineAttributes,
		newDB->pointAttributes);

	faceDrawIndirects.Insert(mvk::DrawIndexedIndirectCommand{});
	edgeDrawIndirects.Insert(mvk::DrawIndirectCommand{});
	nodeDrawIndirects.Insert(mvk::DrawIndirectCommand{});

	bvhTree->AddTree(faceIndices, nodeBuffers.GetDeviceInfo(), GetNodes());
	bvhTree->SetOffset(static_cast<IndexType>(faceIndices.GetSize() - 1), { 0.1, 0.1, 0.1 });
}

void DeviceMeshContainer::OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB)
{
	faceIndices.Set(index, newDB->faceIndices);
	edgeIndices.Set(index, newDB->edgeIndices);

	const auto range = std::views::iota(
		static_cast<IndexType>(0),
		static_cast<IndexType>(newDB->nodes.size()));
	std::vector<IndexType> indices(range.begin(), range.end());
	nodeIndices.Set(index, indices);

	nodeBuffers.Set(
		index,
		newDB->nodes,
		newDB->normals,
		newDB->triangleAttributes,
		newDB->lineAttributes,
		newDB->pointAttributes);

	bvhTree->SetTree(index, faceIndices, nodeBuffers.GetDeviceInfo(), GetNodes());
}

void DeviceMeshContainer::OnDeleteDB(IndexType index, const DB* prevDB)
{
	faceIndices.Remove(index);
	edgeIndices.Remove(index);
	nodeIndices.Remove(index);
	nodeBuffers.Remove(index);

	faceDrawIndirects.Remove(index);
	edgeDrawIndirects.Remove(index);
	nodeDrawIndirects.Remove(index);

	bvhTree->RemoveTree(index);
}
