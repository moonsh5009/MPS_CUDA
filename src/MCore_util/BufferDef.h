#pragma once

#include "TypeDef.h"

#include <glm/glm.hpp>

namespace mvk
{
	namespace vbo
	{
		struct alignas(4) PointAttribute
		{
			alignas(4)	glm::u8vec4 color;
			alignas(4)	float size;

			friend std::ostream& operator<<(std::ostream& os, const PointAttribute& attr)
			{
				return os << "PointAttribute{color:("
					<< static_cast<int>(attr.color.r) << ","
					<< static_cast<int>(attr.color.g) << ","
					<< static_cast<int>(attr.color.b) << ","
					<< static_cast<int>(attr.color.a) << "), size:"
					<< attr.size << "}";
			}
		};

		struct alignas(4) LineAttribute
		{
			alignas(4)	glm::u8vec4 color;
			alignas(4)	float thickness;

			friend std::ostream& operator<<(std::ostream& os, const LineAttribute& attr)
			{
				return os << "LineAttribute{color:("
					<< static_cast<int>(attr.color.r) << ","
					<< static_cast<int>(attr.color.g) << ","
					<< static_cast<int>(attr.color.b) << ","
					<< static_cast<int>(attr.color.a) << "), thickness:"
					<< attr.thickness << "}";
			}
		};

		struct alignas(8) TriangleAttribute
		{
			alignas(4)	glm::u8vec4 color;
			alignas(8)	glm::vec2 texCoord;

			friend std::ostream& operator<<(std::ostream& os, const TriangleAttribute& attr)
			{
				return os << "TriangleAttribute{color:("
					<< static_cast<int>(attr.color.r) << ","
					<< static_cast<int>(attr.color.g) << ","
					<< static_cast<int>(attr.color.b) << ","
					<< static_cast<int>(attr.color.a) << "), texCoord:("
					<< attr.texCoord.x << "," << attr.texCoord.y << ")}";
			}
		};
	}

	struct alignas(4) DrawIndirectCommand
	{
		alignas(4) uint32_t vertexCount;
		alignas(4) uint32_t instanceCount;
		alignas(4) uint32_t firstVertex;
		alignas(4) uint32_t firstInstance;

		friend std::ostream& operator<<(std::ostream& os, const DrawIndirectCommand& cmd)
		{
			return os << "DrawIndirect{vertexCount:" << cmd.vertexCount
				<< ", instanceCount:" << cmd.instanceCount
				<< ", firstVertex:" << cmd.firstVertex
				<< ", firstInstance:" << cmd.firstInstance << "}";
		}
	};
	struct alignas(4) DrawIndexedIndirectCommand
	{
		alignas(4) uint32_t indexCount;
		alignas(4) uint32_t instanceCount;
		alignas(4) uint32_t firstIndex;
		alignas(4) int32_t vertexOffset;
		alignas(4) uint32_t firstInstance;

		friend std::ostream& operator<<(std::ostream& os, const DrawIndexedIndirectCommand& cmd)
		{
			return os << "DrawIndexedIndirect{indexCount:" << cmd.indexCount
				<< ", instanceCount:" << cmd.instanceCount
				<< ", firstIndex:" << cmd.firstIndex
				<< ", vertexOffset:" << cmd.vertexOffset
				<< ", firstInstance:" << cmd.firstInstance << "}";
		}
	};
}