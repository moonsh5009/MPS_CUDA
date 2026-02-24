#include "stdafx.h"
#include "DeviceClothContainer.h"

#include <unordered_map>

REGISTRY_DEVICE_CONTAINER(DeviceClothContainer)

namespace
{
	struct PairHash
	{
		size_t operator()(const std::pair<IndexType, IndexType>& p) const
		{
			return std::hash<uint64_t>{}(static_cast<uint64_t>(p.first) << 32 | p.second);
		}
	};

	std::pair<IndexType, IndexType> MakeEdgeKey(IndexType a, IndexType b)
	{
		return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
	}

	std::vector<double> ComputeRestLengths(
		const std::vector<IndexType>& edgeIndices,
		const std::vector<mcore::Vector3>& nodes)
	{
		const auto edgeCount = edgeIndices.size() / 2;
		std::vector<double> lengths(edgeCount);
		for (size_t i = 0; i < edgeCount; ++i)
		{
			const auto v0 = edgeIndices[i * 2 + 0];
			const auto v1 = edgeIndices[i * 2 + 1];
			lengths[i] = (nodes[v1] - nodes[v0]).norm();
		}
		return lengths;
	}

	std::vector<BendEdge> BuildBendEdges(
		const std::vector<IndexType>& faceIndices,
		size_t nodeCount)
	{
		const auto faceCount = faceIndices.size() / 3;

		struct FaceInfo
		{
			IndexType faceIndex;
			IndexType oppositeVertex;
		};

		std::unordered_map<std::pair<IndexType, IndexType>, std::vector<FaceInfo>, PairHash> edgeFaceMap;
		edgeFaceMap.reserve(faceCount * 3);

		for (size_t f = 0; f < faceCount; ++f)
		{
			const IndexType tri[3] = {
				faceIndices[f * 3 + 0],
				faceIndices[f * 3 + 1],
				faceIndices[f * 3 + 2]
			};

			for (int e = 0; e < 3; ++e)
			{
				const auto va = tri[e];
				const auto vb = tri[(e + 1) % 3];
				const auto vc = tri[(e + 2) % 3];
				const auto key = MakeEdgeKey(va, vb);
				edgeFaceMap[key].push_back({ static_cast<IndexType>(f), vc });
			}
		}

		std::vector<BendEdge> edges;
		edges.reserve(edgeFaceMap.size());

		for (const auto& [edgeKey, faces] : edgeFaceMap)
		{
			if (faces.size() == 2)
			{
				BendEdge be;
				be.v0 = edgeKey.first;
				be.v1 = edgeKey.second;
				be.v2 = faces[0].oppositeVertex;
				be.v3 = faces[1].oppositeVertex;
				edges.push_back(be);
			}
		}

		return edges;
	}
}

DeviceClothContainer::DeviceClothContainer(ISimulateManager* pSimulateManager)
	: simulate::DeviceContainer<ClothData>{ pSimulateManager }
	, mesh{ pSimulateManager }
	, kinetic{ pSimulateManager }
{}

void DeviceClothContainer::OnAddDB(const DB* newDB)
{
	const auto meshIndex = mesh->GetIndex(newDB->mesh.GetKey());
	const auto kineticIndex = kinetic->GetIndex(newDB->kinetic.GetKey());

	meshIndices.Insert(meshIndex);
	kineticIndices.Insert(kineticIndex);
	hostMeshIndices.Insert(meshIndex);
	hostKineticIndices.Insert(kineticIndex);

	stretchStiffnessArray.Insert(newDB->stretchStiffness);
	bendStiffnessArray.Insert(newDB->bendStiffness);
	dampingCoeffArray.Insert(newDB->dampingCoeff);
	cgMaxIterationsArray.Insert(newDB->cgMaxIterations);

	const auto lengths = ComputeRestLengths(newDB->mesh->edgeIndices, newDB->mesh->nodes);
	restLengths.Insert(lengths);

	const auto bEdges = BuildBendEdges(newDB->mesh->faceIndices, newDB->mesh->nodes.size());
	bendEdges.Insert(bEdges);

	activeNodeCount += newDB->mesh->nodes.size();
	activeEdgeCount += newDB->mesh->edgeIndices.size() / 2;
	activeBendEdgeCount += bEdges.size();
}

void DeviceClothContainer::OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB)
{
	const auto meshIndex = mesh->GetIndex(newDB->mesh.GetKey());
	const auto kineticIndex = kinetic->GetIndex(newDB->kinetic.GetKey());

	meshIndices.Set(index, meshIndex);
	kineticIndices.Set(index, kineticIndex);
	hostMeshIndices.Set(index, meshIndex);
	hostKineticIndices.Set(index, kineticIndex);

	stretchStiffnessArray.Set(index, newDB->stretchStiffness);
	bendStiffnessArray.Set(index, newDB->bendStiffness);
	dampingCoeffArray.Set(index, newDB->dampingCoeff);
	cgMaxIterationsArray.Set(index, newDB->cgMaxIterations);

	const auto prevBE = BuildBendEdges(prevDB->mesh->faceIndices, prevDB->mesh->nodes.size());
	activeNodeCount -= prevDB->mesh->nodes.size();
	activeEdgeCount -= prevDB->mesh->edgeIndices.size() / 2;
	activeBendEdgeCount -= prevBE.size();

	const auto lengths = ComputeRestLengths(newDB->mesh->edgeIndices, newDB->mesh->nodes);
	restLengths.Set(index, lengths);

	const auto bEdges = BuildBendEdges(newDB->mesh->faceIndices, newDB->mesh->nodes.size());
	bendEdges.Set(index, bEdges);

	activeNodeCount += newDB->mesh->nodes.size();
	activeEdgeCount += newDB->mesh->edgeIndices.size() / 2;
	activeBendEdgeCount += bEdges.size();
}

void DeviceClothContainer::OnDeleteDB(IndexType index, const DB* prevDB)
{
	const auto prevBE = BuildBendEdges(prevDB->mesh->faceIndices, prevDB->mesh->nodes.size());
	activeNodeCount -= prevDB->mesh->nodes.size();
	activeEdgeCount -= prevDB->mesh->edgeIndices.size() / 2;
	activeBendEdgeCount -= prevBE.size();

	meshIndices.Remove(index);
	kineticIndices.Remove(index);
	hostMeshIndices.Remove(index);
	hostKineticIndices.Remove(index);

	stretchStiffnessArray.Remove(index);
	bendStiffnessArray.Remove(index);
	dampingCoeffArray.Remove(index);
	cgMaxIterationsArray.Remove(index);

	restLengths.Remove(index);
	bendEdges.Remove(index);
}
