#pragma once

#include "DBRegistery.h"

struct MeshData : public database::Data
{
	DATABASE_FIELD(MeshData, database::Data, (
		((std::vector<IndexType>), faceIndices, "FaceIndices"),
		((std::vector<IndexType>), edgeIndices, "EdgeIndices"),
		((std::vector<mcore::Vector3>), nodes, "Nodes"),
		((std::vector<mcore::Vector3>), normals, "Normals"),
		((std::vector<mvk::vbo::TriangleAttribute>), triangleAttributes, "TriangleAttributes"),
		((std::vector<mvk::vbo::LineAttribute>), lineAttributes, "LineAttributes"),
		((std::vector<mvk::vbo::PointAttribute>), pointAttributes, "PointAttributes")
		), MESH);

	void Initialize(const IDBSession* pDBSession) override
	{
		Parent::Initialize(pDBSession);

		faceIndices.clear();
		edgeIndices.clear();
		nodes.clear();
		normals.clear();
		triangleAttributes.clear();
		pointAttributes.clear();
		lineAttributes.clear();
	}
};