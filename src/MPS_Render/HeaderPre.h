#ifndef __HEADERPRE_H__
#define __HEADERPRE_H__

#ifndef __MY_EXT_CLASS__

#ifndef BUILD_MPS_RENDER

#pragma comment(lib, "MPS_Render.lib")
#define __MY_EXT_CLASS__	__declspec(dllimport)
#define __MY_EXT_API__		__declspec(dllimport)

#else

#define __MY_EXT_CLASS__	__declspec(dllexport)
#define __MY_EXT_API__		__declspec(dllexport)

#endif

#endif

#endif