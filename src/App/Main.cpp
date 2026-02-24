#include "stdafx.h"
#include <windows.h>

#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#include "../MCore_util/DllLoader.h"
#include "../MCore_util/WindowEventInfo.h"

#include "../MCore_system/System.h"
#include "../MPS_system/PhysicsSimulateManager.h"

#include "../MPS_database/MeshPool.h"
#include "../MPS_database/KineticPool.h"
#include "../MPS_database/ClothDynamicsPool.h"
#include "../MPS_database/DynamicsSystemPool.h"
#include "../MPS_database/ForceDynamicsPool.h"
#include "../MPS_database/MeshLoader.h"

#include <string>

std::shared_ptr<mcore::system::System> pSystem;
mcore::IScene* pScene = nullptr;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    auto GetMousePoint = [&](LPARAM lParam) -> std::pair<int, int>
    {
        return { ((int)(short)LOWORD(lParam)), ((int)(short)HIWORD(lParam)) };
    };

    //OutputDebugString((std::to_wstring(uMsg) + L"\n").c_str());
    switch (uMsg)
    {
        //case WM_NCLBUTTONDOWN:
        //    //return 0;
        //    break;

    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        if (pScene)
        {
            pScene->GetUserInputHandler()->OnDraw();
        }

        EndPaint(hwnd, &ps);
        return 0;
    }

    case WM_SIZE:
        if (pScene && wParam != SIZE_MINIMIZED)
        {
            UINT width = LOWORD(lParam);
            UINT height = HIWORD(lParam);

            pScene->GetUserInputHandler()->OnResize(width, height);
        }
        return 0;

    case WM_ERASEBKGND:
        return 0;

    case WM_LBUTTONDOWN:
        if (pScene)
        {
            const auto info = CreateMouseEventInfo(wParam, lParam, mcore::MouseButtonBits::Left);
            pScene->GetUserInputHandler()->OnMouseDown(info);
        }
        break;
    case WM_LBUTTONUP:
        if (pScene)
        {
            const auto info = CreateMouseEventInfo(wParam, lParam, mcore::MouseButtonBits::Left);
            pScene->GetUserInputHandler()->OnMouseUp(info);
        }
        break;
    case WM_RBUTTONDOWN:
        if (pScene)
        {
            const auto info = CreateMouseEventInfo(wParam, lParam, mcore::MouseButtonBits::Right);
            pScene->GetUserInputHandler()->OnMouseDown(info);
        }
        break;
    case WM_RBUTTONUP:
        if (pScene)
        {
            const auto info = CreateMouseEventInfo(wParam, lParam, mcore::MouseButtonBits::Right);
            pScene->GetUserInputHandler()->OnMouseUp(info);
        }
        break;
    case WM_MBUTTONDOWN:
        if (pScene)
        {
            const auto info = CreateMouseEventInfo(wParam, lParam, mcore::MouseButtonBits::Middle);
            pScene->GetUserInputHandler()->OnMouseDown(info);
            SetCapture(hwnd);
        }
        break;
    case WM_MBUTTONUP:
        if (pScene)
        {
            const auto info = CreateMouseEventInfo(wParam, lParam, mcore::MouseButtonBits::Middle);
            pScene->GetUserInputHandler()->OnMouseUp(info);
            ReleaseCapture();
        }
        break;
    case WM_MOUSEMOVE:
        if (pScene)
        {
            const auto info = mcore::CreateMouseMoveEventInfo(wParam, lParam);
            pScene->GetUserInputHandler()->OnMouseMove(info);
        }
        return 0;

    case WM_MOUSEWHEEL:
        if (pScene)
        {
            const auto info = mcore::CreateMouseWheelEventInfo(hwnd, wParam, lParam);
            if (pScene->GetUserInputHandler()->OnMouseWheel(info))
            {
                return 0; // 처리됨
            }
        }
        break;

    case WM_KEYDOWN:
        if (pScene)
        {
            const auto info = CreateKeyEventInfo(wParam, lParam);
            pScene->GetUserInputHandler()->OnKeyDown(info);

            if (info.keyCode == 'A')
            {
                pSystem->GetDBSession()->TransactionGuard([](mcore::IDBSession* pSession)
                {
                    const auto pMeshPool = pSession->GetCastPool<MeshPool>();
                    const auto pKineticPool = pSession->GetCastPool<KineticPool>();
                    const auto pForceDynamicsPool = pSession->GetCastPool<ForceDynamicsPool>();

                    const auto meshKey = [&]
                    {
                        MeshLoader loader(pSystem->GetDBSession());
                        auto pMesh = pMeshPool->NewData();
                        AABB<double> aabb;
                        aabb.Initialize();
                        aabb += {-50., -50., -50. };
                        aabb += { 50., 50., 50. };
                        loader.LoadOBJ("../../../obj/feifei_2.obj", pMesh, aabb);
                        return pMeshPool->Insert(std::move(pMesh));
                    }();

                    const auto pMesh = pMeshPool->Get(meshKey);
                    auto pKinetic = pKineticPool->NewData();
                    pKinetic->masses = std::vector<double>(pMesh->nodes.size(), 1.0);
                    pKinetic->invMasses = std::vector<double>(pMesh->nodes.size(), 1.0);
                    pKinetic->velocities = std::vector<mcore::Vector3>(pMesh->nodes.size(), mcore::Vector3::Zero());
                    pKinetic->forces = std::vector<mcore::Vector3>(pMesh->nodes.size(), mcore::Vector3::Zero());
                    const auto kineticKey = pKineticPool->Insert(std::move(pKinetic));

                    auto pforcedynamics = pForceDynamicsPool->NewData();
                    pforcedynamics->mesh.SetKey(meshKey);
                    pforcedynamics->kinetic.SetKey(kineticKey);
                    const auto forcedynamicskey = pForceDynamicsPool->Insert(std::move(pforcedynamics));
                    return true;

                    return true;
                });
            }
            else if (info.keyCode == 'B')
            {
                MeshLoader loader(pSystem->GetDBSession());
                pSystem->GetDBSession()->TransactionGuard([&loader](mcore::IDBSession* pSession)
                {
                    const auto pMeshPool = pSession->GetCastPool<MeshPool>();
                    const auto pKineticPool = pSession->GetCastPool<KineticPool>();
                    const auto pDynSysPool = pSession->GetCastPool<DynamicsSystemPool>();
                    const auto pClothDynPool = pSession->GetCastPool<ClothDynamicsPool>();

                    auto pMesh = pMeshPool->NewData();
                    AABB<double> aabb;
                    aabb.Initialize();
                    aabb += {-50., -50., -50. };
                    aabb += { 50., 50., 50. };
                    loader.LoadOBJ("../../../obj/LR_cloth.obj", pMesh, aabb);
                    const auto nodeCount = pMesh->nodes.size();
                    const auto meshKey = pMeshPool->Insert(std::move(pMesh));

                    auto pKinetic = pKineticPool->NewData();
                    pKinetic->masses = std::vector<double>(nodeCount, 1.0);
                    pKinetic->invMasses = std::vector<double>(nodeCount, 1.0);
                    pKinetic->velocities = std::vector<mcore::Vector3>(nodeCount, mcore::Vector3::Zero());
                    pKinetic->forces = std::vector<mcore::Vector3>(nodeCount, mcore::Vector3::Zero());
                    pKinetic->fixeds = std::vector<IndexType>(nodeCount, 0);
                    pKinetic->fixeds[0] = 1;
                    const auto kineticKey = pKineticPool->Insert(std::move(pKinetic));

                    auto pDynSys = pDynSysPool->NewData();
                    pDynSys->mesh.SetKey(meshKey);
                    pDynSys->kinetic.SetKey(kineticKey);
                    const auto dynSysKey = pDynSysPool->Insert(std::move(pDynSys));

                    auto pCloth = pClothDynPool->NewData();
                    pCloth->dynamicsSystem.SetKey(dynSysKey);
                    pClothDynPool->Insert(std::move(pCloth));

                    return true;
                });
            }
            else if (info.keyCode == 'Z')
            {
                pSystem->GetDBSession()->Undo();
            }
        }
        return 0;

    case WM_KEYUP:
        if (pScene)
        {
            const auto info = CreateKeyEventInfo(wParam, lParam);
            pScene->GetUserInputHandler()->OnKeyUp(info);
        }
        return 0;

    case WM_SETFOCUS:
        if (pScene)
        {
        }
        break;

    case WM_KILLFOCUS:
        if (pScene)
        {
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        break;
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    const wchar_t CLASS_NAME[] = L"VkModernWindow";

    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

    if (!RegisterClassEx(&wc))
    {
        MessageBox(NULL, L"Window registration failed!", L"Error", MB_ICONERROR);
        return 0;
    }

    HWND hwnd = CreateWindowEx(
        WS_EX_APPWINDOW,
        CLASS_NAME, L"Modern & Fast Window",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 900, 700,
        nullptr, nullptr, hInstance, nullptr);

    if (!hwnd)
    {
        MessageBox(NULL, L"Window creation failed!", L"Error", MB_ICONERROR);
        return 0;
    }

    MARGINS margins = { -1 };
    DwmExtendFrameIntoClientArea(hwnd, &margins);

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    try
    {
        auto& manager = mcore::util::DllManager::Instance();
        auto loader1 = manager.LoadDll("MPS_database.dll");
        auto loader2 = manager.LoadDll("MPS_simulate.dll");
        auto loader3 = manager.LoadDll("MPS_system.dll");

        pSystem = std::make_shared<mcore::system::System>();
        pSystem->SetSimulateManager(std::make_unique<PhysicsSimulateManager>());
        pSystem->Initialize(hwnd);

        auto pRenderCore = pSystem->GetRenderCore();
        pRenderCore->AddScene(hwnd);
        pScene = pRenderCore->GetScene(hwnd);
    }
    catch (const std::exception& e)
    {
        MessageBoxA(NULL, e.what(), "Vulkan Init Error", MB_ICONERROR);
        return 0;
    }

    MSG msg = {};
    BOOL running = TRUE;
    while (running)
    {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = FALSE;
                pScene = nullptr;
                pSystem.reset();
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (pSystem)
        {
            pSystem->Run();
        }
        InvalidateRect(hwnd, NULL, FALSE);
        Sleep(1);
    }

    pScene = nullptr;
    pSystem.reset();

    return (int)msg.wParam;
}