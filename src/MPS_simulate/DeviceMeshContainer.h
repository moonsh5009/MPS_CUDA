#pragma once

#include "../MCore_simulate/DeviceReference.h"

#include "../MPS_database/MeshData.h"
#include "../MCore_simulate/TriangleBVHTree.h"

#include "HeaderPre.h"

struct DeviceMeshData
{
	simulate::DeviceObjectData info;
	mcuda::DeviceMultiArrayInfo faceInfo;
	mcuda::DeviceMultiArrayInfo edgeInfo;
	mcuda::DeviceMultiArrayInfo nodeInfo;

	IndexType* faces;
	IndexType* edges;
	mcore::Vector3* nodes;
	mcore::Vector3* normals;

	mvk::vbo::TriangleAttribute* triangleAttributes;
	mvk::vbo::LineAttribute* lineAttributes;
	mvk::vbo::PointAttribute* pointAttributes;

	mvk::DrawIndexedIndirectCommand* faceDrawIndirects;
};

class __MY_EXT_CLASS__ DeviceMeshContainer : public simulate::DeviceContainer<MeshData>
{
public:
	using DeviceData = DeviceMeshData;

	DeviceMeshContainer(ISimulateManager* pSimulateManager);

	void OnAddDB(const DB* newDB) override;
	void OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB) override;
	void OnDeleteDB(IndexType index, const DB* prevDB) override;

	DeviceData GetDeviceData() const
	{
		DeviceData data;
		data.info = simulate::DeviceContainer<MeshData>::GetDeviceData();
		data.faceInfo = faceIndices.GetDeviceInfo();
		data.edgeInfo = edgeIndices.GetDeviceInfo();
		data.nodeInfo = nodeBuffers.GetDeviceInfo();
		data.faces = faceIndices.GetBuffer().GetData();
		data.edges = edgeIndices.GetBuffer().GetData();
		data.nodes = GetNodes().GetData();
		data.normals = GetNormals().GetData();
		data.triangleAttributes = GetTriangleAttributes().GetData();
		data.lineAttributes = GetLineAttributes().GetData();
		data.pointAttributes = GetPointAttributes().GetData();
		data.faceDrawIndirects = faceDrawIndirects.GetBuffer().GetData();
		return data;
	}

	mcuda::VKDeviceBuffer<mcore::Vector3>& GetNodes() { return nodeBuffers.GetBuffer<0>(); }
	mcuda::VKDeviceBuffer<mcore::Vector3>& GetNormals() { return nodeBuffers.GetBuffer<1>(); }
	mcuda::VKDeviceBuffer<mvk::vbo::TriangleAttribute>& GetTriangleAttributes() { return nodeBuffers.GetBuffer<2>(); }
	mcuda::VKDeviceBuffer<mvk::vbo::LineAttribute>& GetLineAttributes() { return nodeBuffers.GetBuffer<3>(); }
	mcuda::VKDeviceBuffer<mvk::vbo::PointAttribute>& GetPointAttributes() { return nodeBuffers.GetBuffer<4>(); }

	const mcuda::VKDeviceBuffer<mcore::Vector3>& GetNodes() const { return nodeBuffers.GetBuffer<0>(); }
	const mcuda::VKDeviceBuffer<mcore::Vector3>& GetNormals() const { return nodeBuffers.GetBuffer<1>(); }
	const mcuda::VKDeviceBuffer<mvk::vbo::TriangleAttribute>& GetTriangleAttributes() const { return nodeBuffers.GetBuffer<2>(); }
	const mcuda::VKDeviceBuffer<mvk::vbo::LineAttribute>& GetLineAttributes() const { return nodeBuffers.GetBuffer<3>(); }
	const mcuda::VKDeviceBuffer<mvk::vbo::PointAttribute>& GetPointAttributes() const { return nodeBuffers.GetBuffer<4>(); }

	mcuda::VKDeviceMultiArray<IndexType> faceIndices;
	mcuda::VKDeviceMultiArray<IndexType> edgeIndices;
	mcuda::VKDeviceMultiArray<IndexType> nodeIndices;
	mcuda::VKDeviceMultiArray<
		mcore::Vector3,
		mcore::Vector3,
		mvk::vbo::TriangleAttribute,
		mvk::vbo::LineAttribute,
		mvk::vbo::PointAttribute> nodeBuffers;

	mcuda::VKDeviceSingleArray<mvk::DrawIndexedIndirectCommand> faceDrawIndirects;
	mcuda::VKDeviceSingleArray<mvk::DrawIndirectCommand> edgeDrawIndirects;
	mcuda::VKDeviceSingleArray<mvk::DrawIndirectCommand> nodeDrawIndirects;

	std::unique_ptr<simulate::TriangleBVHTree> bvhTree;
};

#include "HeaderPost.h"
