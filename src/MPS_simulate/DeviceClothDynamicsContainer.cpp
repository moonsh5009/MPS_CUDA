#include "stdafx.h"
#include "DeviceClothDynamicsContainer.h"

#include <unordered_map>

REGISTRY_DEVICE_CONTAINER(DeviceClothDynamicsContainer)

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

DeviceClothDynamicsContainer::DeviceClothDynamicsContainer(ISimulateManager* pSimulateManager)
	: simulate::DeviceContainer<ClothDynamics>{ pSimulateManager }
	, dynamicsSystem{ pSimulateManager }
{}

void DeviceClothDynamicsContainer::OnAddDB(const DB* newDB)
{
	const auto dynSysIndex = dynamicsSystem->GetIndex(newDB->dynamicsSystem.GetKey());

	dynamicsSystemIndices.Insert(dynSysIndex);
	hostDynamicsSystemIndices.Insert(dynSysIndex);

	stretchStiffnessArray.Insert(newDB->stretchStiffness);
	bendStiffnessArray.Insert(newDB->bendStiffness);
	dampingCoeffArray.Insert(newDB->dampingCoeff);
	hostStretchStiffnessArray.Insert(newDB->stretchStiffness);
	hostBendStiffnessArray.Insert(newDB->bendStiffness);
	hostDampingCoeffArray.Insert(newDB->dampingCoeff);

	const auto& meshRef = newDB->dynamicsSystem->mesh;
	const auto lengths = ComputeRestLengths(meshRef->edgeIndices, meshRef->nodes);
	restLengths.Insert(lengths);

	const auto bEdges = BuildBendEdges(meshRef->faceIndices, meshRef->nodes.size());
	bendEdges.Insert(bEdges);

	activeNodeCount += meshRef->nodes.size();
	activeEdgeCount += meshRef->edgeIndices.size() / 2;
	activeBendEdgeCount += bEdges.size();
}

void DeviceClothDynamicsContainer::OnModifyDB(IndexType index, const DB* prevDB, const DB* newDB)
{
	const auto dynSysIndex = dynamicsSystem->GetIndex(newDB->dynamicsSystem.GetKey());

	dynamicsSystemIndices.Set(index, dynSysIndex);
	hostDynamicsSystemIndices.Set(index, dynSysIndex);

	stretchStiffnessArray.Set(index, newDB->stretchStiffness);
	bendStiffnessArray.Set(index, newDB->bendStiffness);
	dampingCoeffArray.Set(index, newDB->dampingCoeff);
	hostStretchStiffnessArray.Set(index, newDB->stretchStiffness);
	hostBendStiffnessArray.Set(index, newDB->bendStiffness);
	hostDampingCoeffArray.Set(index, newDB->dampingCoeff);

	const auto& prevMeshRef = prevDB->dynamicsSystem->mesh;
	const auto prevBE = BuildBendEdges(prevMeshRef->faceIndices, prevMeshRef->nodes.size());
	activeNodeCount -= prevMeshRef->nodes.size();
	activeEdgeCount -= prevMeshRef->edgeIndices.size() / 2;
	activeBendEdgeCount -= prevBE.size();

	const auto& meshRef = newDB->dynamicsSystem->mesh;
	const auto lengths = ComputeRestLengths(meshRef->edgeIndices, meshRef->nodes);
	restLengths.Set(index, lengths);

	const auto bEdges = BuildBendEdges(meshRef->faceIndices, meshRef->nodes.size());
	bendEdges.Set(index, bEdges);

	activeNodeCount += meshRef->nodes.size();
	activeEdgeCount += meshRef->edgeIndices.size() / 2;
	activeBendEdgeCount += bEdges.size();
}

void DeviceClothDynamicsContainer::OnDeleteDB(IndexType index, const DB* prevDB)
{
	const auto& prevMeshRef = prevDB->dynamicsSystem->mesh;
	const auto prevBE = BuildBendEdges(prevMeshRef->faceIndices, prevMeshRef->nodes.size());
	activeNodeCount -= prevMeshRef->nodes.size();
	activeEdgeCount -= prevMeshRef->edgeIndices.size() / 2;
	activeBendEdgeCount -= prevBE.size();

	dynamicsSystemIndices.Remove(index);
	hostDynamicsSystemIndices.Remove(index);

	stretchStiffnessArray.Remove(index);
	bendStiffnessArray.Remove(index);
	dampingCoeffArray.Remove(index);
	hostStretchStiffnessArray.Remove(index);
	hostBendStiffnessArray.Remove(index);
	hostDampingCoeffArray.Remove(index);

	restLengths.Remove(index);
	bendEdges.Remove(index);
}
