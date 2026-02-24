#pragma once

#include "../MCore_util/Generator.h"

#include "../MCore_interface/DBDef.h"

#include <memory>
#include <set>

namespace mcore::database
{
	class KeyGenerator
	{
	public:
		KeyGenerator()
			: m_currentKey{ 1 }
		{}

		DBKey NewKey()
		{
			if (m_deletedKeys.empty())
			{
				return m_currentKey++;
			}

			const auto it = m_deletedKeys.begin();
			const auto reused = *it;
			m_deletedKeys.erase(it);
			return reused;
		}

		void InsertKey(DBKey key)
		{
			if (const auto itr = m_deletedKeys.find(key); itr != m_deletedKeys.end())
			{
				m_deletedKeys.erase(itr);
				return;
			}

			for (; m_currentKey <= key; ++m_currentKey)
			{
				m_deletedKeys.insert(m_currentKey);
			}
		}

		void DeleteKey(DBKey key)
		{
			m_deletedKeys.insert(key);
		}

	private:
		mcore::DBKey m_currentKey;
		std::set<DBKey> m_deletedKeys;
	};
}