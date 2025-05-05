#pragma once
#include <chrono>
#include <unordered_map>
#include <sstream>
#include <string>
#include <debugapi.h>

#define CHRRONO_TIME_TEST

namespace mcuda
{
	namespace util
	{
		using TimePoint = std::chrono::system_clock::time_point;
		using TimeDuration = std::chrono::duration<double, std::milli>;

		namespace MTimer
		{
		#ifdef CHRRONO_TIME_TEST
			static TimePoint startPoint;
			static std::unordered_map<std::string_view, TimePoint> mapStartPoint;
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
				TimeDuration d = std::chrono::system_clock::now() - startPoint;

				std::stringstream ss;
				ss << d.count() << " ms" << std::endl;
				OutputDebugStringA(ss.str().c_str());
			#endif
			}

			static void End(std::string_view key)
			{
			#ifdef CHRRONO_TIME_TEST
				const auto t = mapStartPoint[key];
				mapStartPoint.erase(mapStartPoint.find(key));
				TimeDuration d = std::chrono::system_clock::now() - t;

				std::stringstream ss;
				ss << key << " : " << d.count() << " ms" << std::endl;
				OutputDebugStringA(ss.str().c_str());
			#endif
			}

			static void EndWithMessage(std::string_view msg)
			{
			#ifdef CHRRONO_TIME_TEST
				TimeDuration d = std::chrono::system_clock::now() - startPoint;

				std::stringstream ss;
				ss << d.count() << " ms" << ", " << msg << std::endl;
				OutputDebugStringA(ss.str().c_str());
			#endif
			}

			static void EndWithMessage(std::string_view key, std::string_view msg)
			{
			#ifdef CHRRONO_TIME_TEST
				const auto t = mapStartPoint[key];
				mapStartPoint.erase(mapStartPoint.find(key));
				TimeDuration d = std::chrono::system_clock::now() - t;

				std::stringstream ss;
				ss << key << " : " << d.count() << " ms" << ", " << msg << std::endl;
				OutputDebugStringA(ss.str().c_str());
			#endif
			}
		};
	}
}