#include "stdafx.h"
#include "MeshLoader.h"

#include "MeshPool.h"

#include <fstream>
#include <sstream>
#include <unordered_map>

MeshLoader::MeshLoader(mcore::IDBSession* pSession)
	: m_pSession{ pSession }
{}

void MeshLoader::LoadOBJ(
	const std::string& filename,
	const std::shared_ptr<MeshData>& pMeshData,
	const AABB<double>& aabb)
{
	const auto pMeshPool = m_pSession->GetCastPool<MeshPool>();

	std::ifstream file(filename);
	if (!file.is_open())
		throw std::runtime_error("Failed to open OBJ file: " + filename);

	struct PairHash
	{
		std::size_t operator()(const std::pair<IndexType, IndexType>& p) const noexcept
		{
			return std::hash<IndexType>{}(p.first) ^ (std::hash<IndexType>{}(p.second) << 1);
		}
	};
	std::unordered_map<std::pair<IndexType, IndexType>, IndexType, PairHash> edgeMap;
	IndexType edgeIndex = 0;

	AABB<double> meshAABB;
	meshAABB.Initialize();

	std::string line;
	while (std::getline(file, line))
	{
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;

		if (prefix == "v")
		{
			REAL x, y, z;
			iss >> x >> y >> z;
			pMeshData->nodes.emplace_back(x, y, z);
			meshAABB += { x, y, z };
		}
		else if (prefix == "f")
		{
			std::vector<IndexType> faceIndices;
			std::string vertex;

			while (iss >> vertex)
			{
				std::istringstream viss(vertex);
				std::string indexStr;
				std::getline(viss, indexStr, '/');

				IndexType vertexIndex = std::stoi(indexStr) - 1;
				faceIndices.push_back(vertexIndex);
			}

			for (const auto idx : faceIndices)
			{
				pMeshData->faceIndices.push_back(idx);
			}

			for (size_t i = 0; i < faceIndices.size(); ++i)
			{
				IndexType v1 = faceIndices[i];
				IndexType v2 = faceIndices[(i + 1) % faceIndices.size()];

				if (v1 > v2)
					std::swap(v1, v2);

				auto edgePair = std::make_pair(v1, v2);
				if (edgeMap.find(edgePair) == edgeMap.end())
				{
					edgeMap[edgePair] = edgeIndex++;
					pMeshData->edgeIndices.push_back(v1);
					pMeshData->edgeIndices.push_back(v2);
				}
			}
		}
	}
	file.close();

	if (meshAABB.IsAvailable() && aabb.IsAvailable())
	{
		const auto meshCenter = meshAABB.GetCenter();
		const auto meshSize = meshAABB.GetMax() - meshAABB.GetMin();
		const auto targetCenter = aabb.GetCenter();
		const auto targetSize = aabb.GetMax() - aabb.GetMin();

		const double scaleX = targetSize.x / meshSize.x;
		const double scaleY = targetSize.y / meshSize.y;
		const double scaleZ = targetSize.z / meshSize.z;
		const double scale = std::min({ scaleX, scaleY, scaleZ });

		for (auto& pos : pMeshData->nodes)
		{
			pos[0] -= meshCenter[0];
			pos[1] -= meshCenter[1];
			pos[2] -= meshCenter[2];
			pos *= scale;
			pos[0] += targetCenter[0];
			pos[1] += targetCenter[1];
			pos[2] += targetCenter[2];
		}
	}

	pMeshData->normals.resize(pMeshData->nodes.size(), mcore::Vector3::UnitZ());
	pMeshData->triangleAttributes.resize(pMeshData->nodes.size(),
		mvk::vbo::TriangleAttribute{ { 34, 200, 162, 255 }, { 0.f, 0.f } });
	pMeshData->lineAttributes.resize(pMeshData->nodes.size(),
		mvk::vbo::LineAttribute{ { 0, 0, 0, 255 }, 2.f });
	pMeshData->pointAttributes.resize(pMeshData->nodes.size(),
		mvk::vbo::PointAttribute{ { 0, 0, 0, 255 }, 3.f });
}