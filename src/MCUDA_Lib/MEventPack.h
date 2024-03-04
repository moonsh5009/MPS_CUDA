#pragma once

#include <vector>
#include <set>
#include <unordered_map>
#include <functional>

namespace mevent
{
	enum class MMouseAction : int
	{
		Up,
		Down,
	};

	enum class MKeyAction : int
	{
		Up,
		Down,
	};

	enum class MMouseButton : int
	{
		LMouse,
		RMouse,
		WMouse,
	};

	enum MMods : int
	{
		None = 0,
		Shift = 2,
		Ctrl = 8,
		Wheel = 16,
	};
}

namespace mgpu
{
	enum MEventSequence
	{
		Prev,
		Norm,
		After,
		Size
	};

	template <class Func>
	class MEventPack final
	{
	public:
		MEventPack() : m_seq{ 0u }
		{
			m_funcPack.resize(MEventSequence::Size);
		}
		~MEventPack() {}

	private:
		uint32_t m_seq;
		std::set<uint32_t> m_seqTmps;
		std::vector<std::unordered_map<uint32_t, Func>> m_funcPack;

	public:
		template <class... A>
		void Notify(A... args)
		{
			for (uint32_t i = 0; i < MEventSequence::Size; i++)
			{
				for (auto& [_, f] : m_funcPack[i])
				{
					f(args...);
				}
			}
		}

		void Connect(Func func, uint32_t key, MEventSequence turn = MEventSequence::Norm)
		{
			m_funcPack[turn].emplace(key, func);
		}
	};
}