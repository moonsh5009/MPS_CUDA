#ifndef __HEADERPRE_H__
#define __HEADERPRE_H__

#ifndef __MY_EXT_CLASS__

#ifndef BUILD_MPS_SIMULATE

#pragma comment(lib, "MPS_simulate.lib")
#define __MY_EXT_CLASS__	__declspec(dllimport)

#else

#define __MY_EXT_CLASS__	__declspec(dllexport)

#endif

#endif

#endif