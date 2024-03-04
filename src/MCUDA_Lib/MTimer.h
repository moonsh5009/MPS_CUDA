#pragma once
#include <chrono>
#include <unordered_map>

typedef std::chrono::system_clock::time_point CTimePoint;
typedef std::chrono::duration<double, std::milli> CTimeDuration;

#define CHRRONO_TIME_TEST

namespace MTimer
{
#ifdef CHRRONO_TIME_TEST
	static CTimePoint startPoint;
	static std::unordered_map<std::string_view, CTimePoint> mapStartPoint;
#endif

	static void Start()
	{
	#ifdef CHRRONO_TIME_TEST
		startPoint = std::chrono::system_clock::now();
	#endif
	}

	static void Start(std::string_view key)
	{
	#ifdef CHRRONO_TIME_TEST
		mapStartPoint.emplace(key, std::chrono::system_clock::now());
	#endif
	}

	static void End()
	{
	#ifdef CHRRONO_TIME_TEST
		CTimeDuration d = std::chrono::system_clock::now() - startPoint;
		std::cout << d.count() << " ms" << std::endl;
	#endif
	}

	static void End(std::string_view key)
	{
	#ifdef CHRRONO_TIME_TEST
		const auto t = mapStartPoint[key];
		mapStartPoint.erase(mapStartPoint.find(key));
		CTimeDuration d = std::chrono::system_clock::now() - t;
		std::cout << key << " : " << d.count() << " ms" << std::endl;// << std::endl;
	#endif
	}
};