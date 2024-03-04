#pragma once

#include "MPSMaterial.h"
#include "MPSObject.h"
#include "MPSTree.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ Model
	{
	public:
		Model() = delete;
		Model(std::unique_ptr<Material>&& pMaterial, std::unique_ptr<Object>&& pObject, std::unique_ptr<Tree>&& pTree);

	public:
		Material* GetMaterial() const { return m_pMaterial.get(); };
		Tree* GetTree() const { return m_pTree.get(); };
		Object* GetTarget() const { return m_pObject.get(); };
		size_t GetTypeID() const { return typeid(*this).hash_code(); };

	protected:
		std::unique_ptr<Material> m_pMaterial;
		std::unique_ptr<Object> m_pObject;
		std::unique_ptr<Tree> m_pTree;
	};
};

#include "HeaderPost.h"