#include "stdafx.h"
#include "MPSSystem.h"

#include "../MPS_Object/MPSGBArchiver.h"

#include "../MPS_Render/MPSRenderManager.h"
#include "../MPS_Render/MPSCamera.h"

#include "../MPS_Object/MPSSPHModel.h"
#include "../MPS_Object/MPSSPHObject.h"
#include "../MPS_Object/MPSSPHMaterial.h"
#include "../MPS_Object/MPSSpatialHash.h"

mps::System::System() : SystemEventHandler{},
    m_frame{ 0 }
{
}

void mps::System::Initalize()
{
    SystemEventHandler::Initalize();

    {
        m_pGBArchiver = std::make_shared<mps::GBArchiver>();
        m_pGBArchiver->Initalize();
        m_pGBArchiver->UpdateLight(mps::LightParam{ glm::dvec3{ 10.0, 10.0, 10.0 }, glm::fvec4{ 1.0f, 1.0f, 1.0f, 1.0f } });
        m_pGBArchiver->m_physicsParam.gravity = REAL3{ 0.0, -9.8, 0.0 };
        m_pGBArchiver->m_physicsParam.dt = 0.01;
    }
    {
        m_pRenderManager = std::make_shared<mps::rndr::RenderManager>();
        m_pRenderManager->Initalize(m_pGBArchiver.get());

        m_pCamera = std::make_shared<mps::rndr::Camera>();
        m_pCameraHandler = std::make_unique<mps::rndr::CameraUserInputEventHandler>(m_pCamera.get());

        m_pSPHModel = std::make_shared<mps::SPHModel>(
            std::make_unique<mps::SPHMaterial>(),
            std::make_unique<mps::SPHObject>(),
            std::make_unique<mps::SpatialHash>());
        ResizeParticle(100);

        m_pRenderManager->AddModel(m_pSPHModel);
    }

    glEnable(GL_DEPTH_TEST);
}

void mps::System::OnWMouseDown(mevent::Flag flag, glm::ivec2 curPos)
{
    m_pCameraHandler->OnWMouseDown(flag, curPos);
}

void mps::System::OnWMouseUp(mevent::Flag flag, glm::ivec2 curPos)
{
    m_pCameraHandler->OnWMouseUp(flag, curPos);
}

void mps::System::OnMouseMove(mevent::Flag flag, glm::ivec2 curPos)
{
    m_pCameraHandler->OnMouseMove(flag, curPos);
}

void mps::System::OnMouseWheel(mevent::Flag flag, glm::ivec2 offset, glm::ivec2 curPos)
{
    m_pCameraHandler->OnMouseWheel(flag, offset, curPos);
}

void mps::System::OnKeyDown(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());
    uint32_t sqNumParticles = static_cast<uint32_t>(powf(static_cast<float>(pSPHObject->GetSize()), 1.0f / 3.0f));
    uint32_t offset = 50;
    switch (key)
    {
    case 219: // '[':
        ResizeParticle((sqNumParticles > offset) ? sqNumParticles - offset : 1);
        break;
    case 221: // ']':
        ResizeParticle(sqNumParticles + offset);
        break;
    default:
        break;
    }
}

void mps::System::OnKeyUp(uint32_t key, uint32_t repCnt, mevent::Flag flag)
{
    switch (key)
    {
    case 'A':
        break;
    case 'S':
        break;
    default:
        break;
    }
}

void mps::System::OnResize(int width, int height)
{
    m_pCamera->GetProjection()->SetAspectRatio(width, height);
}

#include "../MPS_Computer/AdvectUtil.h"
void mps::System::OnUpdate()
{
    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());

    const auto objRes = pSPHObject->GetDeviceResource<mps::ObjectResource>();
    const auto pObjParam = objRes->GetObjectParam().lock();
    if (!pObjParam) return;

    const auto& physParam = m_pGBArchiver->m_physicsParam;

    mps::kernel::InitForce(*pObjParam);
    mps::kernel::ApplyGravity(physParam, *pObjParam);
    mps::kernel::UpdateVelocity(physParam, *pObjParam);
    mps::kernel::UpdatePosition(physParam, *pObjParam);
}

void mps::System::OnDraw()
{
    m_pCamera->Update(m_pGBArchiver.get());
    m_pGBArchiver->UpdateLightPos(m_pCamera->GetTransform()->GetPosition());

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_pRenderManager->Draw(m_pGBArchiver.get());
}

void mps::System::ResizeParticle(size_t size)
{
    auto pSPHObject = static_cast<mps::SPHObject*>(m_pSPHModel->GetTarget());
    auto pSpatialHash = static_cast<mps::SpatialHash*>(m_pSPHModel->GetTree());
    if (pSPHObject->GetSize() == size * size * size)
        return;

    std::cout << pSPHObject->GetSize() << " to ";
    MTimer::Start("ResizeParticle");

    REAL radius = 0.1f;// pSPHObject->GetSize() == 0 ? 0.1f : pSPHObject->m_radius.GetHost()[0];
    REAL offset = 0.001f;
    REAL stride = radius + radius + offset;
    REAL l = stride * (size - 1) + radius + radius;
    REAL startPos = -l * 0.5f;
    REAL mass = 1.0f;

    std::vector<REAL3> h_pos;
    std::vector<REAL> h_radius;

    h_pos.reserve(size * size * size);
    h_radius.reserve(size * size * size);

    for (int z = 0; z < size; z++)
    {
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                REAL3 pos =
                {
                    startPos + x * stride + radius,
                    startPos + y * stride + radius,
                    startPos + z * stride + radius
                };
                h_pos.emplace_back(pos);
                h_radius.emplace_back(radius);
            }
        }
    }

    pSPHObject->Resize(h_radius.size());
    pSPHObject->m_pos.CopyFromHost(h_pos);
    pSPHObject->m_radius.CopyFromHost(h_radius);
    mps::kernel::InitMass(pSPHObject);

    pSpatialHash->SetObjectSize(pSPHObject->GetSize());
    pSpatialHash->SetCeilSize(radius * 4);
    pSpatialHash->SetHashSize({ 64, 64, 64 });

    std::cout << pSPHObject->GetSize() << std::endl;
    MTimer::End("ResizeParticle");
}