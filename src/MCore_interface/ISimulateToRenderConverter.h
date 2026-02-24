#pragma once

#include "DBDef.h"

#include <memory>

namespace mcore
{
	class ISimulateManager;
	class IRenderModelContainer;
	class ISimulateToRenderConverter
	{
	public:
		ISimulateToRenderConverter() = default;
		virtual ~ISimulateToRenderConverter() = default;
		ISimulateToRenderConverter(const ISimulateToRenderConverter&) = default;
		ISimulateToRenderConverter(ISimulateToRenderConverter&&) = default;
		ISimulateToRenderConverter& operator=(const ISimulateToRenderConverter&) = default;
		ISimulateToRenderConverter& operator=(ISimulateToRenderConverter&&) = default;

		virtual DBTypeID GetTypeID() const = 0;

		virtual void Convert(ISimulateManager* pSimulateManager, IRenderModelContainer* pRenderModelContainer) = 0;
	};
	using SimulateToRenderConverterArray = std::vector<std::unique_ptr<ISimulateToRenderConverter>>;
}