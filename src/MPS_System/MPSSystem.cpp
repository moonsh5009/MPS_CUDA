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
        m_pGBArchiver->UpdateLight(mps::LightParam{ glm::fvec3{ 10.0, 10.0f, 10.0f }, glm::fvec4{ 1.0f, 1.0f, 1.0f, 1.0f } });
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
    case 219: // ']':
        ResizeParticle((sqNumParticles > offset) ? sqNumParticles - offset : 1);
        break;
    case 221: // '[':
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

void mps::System::OnUpdate()
{
}

void mps::System::OnDraw()
{
    m_pCamera->Update(m_pGBArchiver.get());
    m_pGBArchiver->UpdateLightPos(m_pCamera->GetTransform()->GetPosition());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

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

    float radius = 0.1f;// pSPHObject->GetSize() == 0 ? 0.1f : pSPHObject->m_radius.GetHost()[0];
    float offset = 0.001f;
    float stride = radius + radius + offset;
    float l = stride * (size - 1) + radius + radius;
    float startPos = -l * 0.5f;
    float mass = 1.0f;

    std::vector<glm::dvec3> h_pos;
    std::vector<float> h_radius;
    //std::vector<float> mass;
    //std::vector<glm::dvec3> vel;
    //std::vector<glm::dvec3> force;

    h_pos.reserve(size * size * size);
    h_radius.reserve(size * size * size);

    for (int z = 0; z < size; z++)
    {
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                glm::vec3 pos =
                {
                    startPos + x * stride + radius,
                    startPos + y * stride + radius,
                    startPos + z * stride + radius
                };
                h_pos.emplace_back(pos);
                h_radius.emplace_back(radius);
                //pSPHObject->m_mass.GetHost().emplace_back(mass);
                //pSPHObject->m_vel.GetHost().emplace_back(0.0f, 0.0f, 0.0f);
                //pSPHObject->m_force.GetHost().emplace_back(0.0f, 0.0f, 0.0f);
            }
        }
    }

    pSPHObject->Resize(h_radius.size());
    pSPHObject->m_pos.CopyFromHost(h_pos);
    pSPHObject->m_radius.CopyFromHost(h_radius);

    pSpatialHash->SetObjectSize(pSPHObject->GetSize());
    pSpatialHash->SetCeilSize(4 * radius);
    pSpatialHash->SetHashSize({ 64, 64, 64 });

    std::cout << pSPHObject->GetSize() << std::endl;
    MTimer::End("ResizeParticle");
}