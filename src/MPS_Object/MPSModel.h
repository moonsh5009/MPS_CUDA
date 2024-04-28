#pragma once

#include <array>
#include "MPSMaterial.h"
#include "MPSObject.h"
#include "MPSTree.h"

namespace mps
{
	class Model
	{
	public:
		Model() = delete;
		Model(std::unique_ptr<Object>&& pMainObject) : m_pMainObject{ std::move(pMainObject) } {}

	public:
		size_t GetTypeID() const { return typeid(*this).hash_code(); }
		template<class T>
		T* GetTarget() const { return static_cast<T*>(m_pMainObject.get()); }

		virtual Object* GetSubObject() const = 0;
		virtual Material* GetMaterial() const = 0;
		virtual Tree* GetTree() const = 0;

		virtual Object* GetSubObject(uint32_t idx) const = 0;
		virtual Material* GetMaterial(uint32_t idx) const = 0;
		virtual Tree* GetTree(uint32_t idx) const = 0;

	protected:
		std::unique_ptr<Object> m_pMainObject;
	};

	template<uint32_t N_SubObject, uint32_t N_Material, uint32_t N_Tree>
	class VirtualModel : public Model
	{
	public:
		VirtualModel() = delete;
		VirtualModel(std::unique_ptr<Object>&& pMainObject) : Model{ std::move(pMainObject) } {}

	public:
		virtual Object* GetSubObject() const override
		{
			if constexpr (N_SubObject == 0u)
				return nullptr;
			else
				return m_aSubObject.front().get();
		}
		virtual Material* GetMaterial() const override
		{
			if constexpr (N_Material == 0u)
				return nullptr;
			else
				return m_aMaterial.front().get();
		}
		virtual Tree* GetTree() const override
		{
			if constexpr (N_Tree == 0u)
				return nullptr;
			else
				return m_aTree.front().get();
		}

		virtual Object* GetSubObject(uint32_t idx) const override
		{
			if (idx >= N_SubObject) return nullptr;
			return m_aSubObject[idx].get();
		}
		virtual Material* GetMaterial(uint32_t idx) const override
		{
			if (idx >= N_Material) return nullptr;
			return m_aMaterial[idx].get();
		}
		virtual Tree* GetTree(uint32_t idx) const override
		{
			if (idx >= N_Tree) return nullptr;
			return m_aTree[idx].get();
		}

	protected:
		void SetSubObject(uint32_t idx, std::unique_ptr<Object>&& pSubObject)
		{
			if (idx >= N_SubObject) return;
			m_aSubObject[idx] = std::move(pSubObject);
		}
		void SetMaterial(uint32_t idx, std::unique_ptr<Material>&& pMaterial)
		{
			if (idx >= N_Material) return;
			m_aMaterial[idx] = std::move(pMaterial);
		}
		void SetTree(uint32_t idx, std::unique_ptr<Tree>&& pTree)
		{
			if (idx >= N_Tree) return;
			m_aTree[idx] = std::move(pTree);
		}

		std::array<std::unique_ptr<Object>, N_SubObject> m_aSubObject;
		std::array<std::unique_ptr<Material>, N_Material> m_aMaterial;
		std::array<std::unique_ptr<Tree>, N_Tree> m_aTree;
	};
};
