#ifndef SCENECONTROLLER_H
#define SCENECONTROLLER_H

#include <af/Mesh.h>
#include <af/MotionGraph.h>
#include <Qt3DCore/qentity.h>
#include <Qt3DCore/qtransform.h>

#include <Qt3DExtras/QCuboidMesh>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QPhongAlphaMaterial>
#include <Qt3DExtras/QPlaneMesh>
#include <Qt3DExtras/QSphereMesh>
#include <Qt3DRender/QMesh>
#include <QtCore/QObject>
#include <string>

class SceneController3d : public QObject {
    Q_OBJECT
public:
    explicit SceneController3d(Qt3DCore::QEntity* rootEntity = nullptr);
    ~SceneController3d();

    void setRootEntity(Qt3DCore::QEntity* rootEntity);
    void setMotionGraph(const MotionGraph& cGraph);
    void setMesh(const Mesh& cMesh);
    void updateView();

    void setMotionNodesVisible(const bool cIsVisible);
    void setMotionRadiusesVisible(const bool cIsVisible);
    void setMeshVerticesVisible(const bool cIsVisible);

    Qt3DCore::QEntity* addCustom(const Vec3f& cPos, const float cRadius, const QColor& cColor, const std::string& cFilePath);

private:
    void addMotionVertex(const Vec3f& cPos, const float cRadius);
    void addMeshVertex(const Vec3f& cPos);
    bool isAllComponentsSet();
    Qt3DCore::QEntity* addSphere(const Vec3f& cPos,
                                 const float cRadius,
                                 const QColor& cColor,
                                 const float cAlpha = 1.f,
                                 const int cSlices  = 3);
    Qt3DCore::QEntity* addCube(const Vec3f& cPos,
                               const float cRadius,
                               const QColor& cColor,
                               const float cAlpha = 1.f,
                               const int cSlices  = 3);
    void setPos(Qt3DCore::QEntity* entity, const Vec3f& cPos);
    void setRadius(Qt3DCore::QEntity* entity, const float cRadius);
    void clear(std::vector<Qt3DCore::QEntity*>& entities);
    void remove(Qt3DCore::QEntity* entity);

    Qt3DCore::QEntity* mRootEntity;

    const MotionGraph* mMotionGraph;
    const Mesh* mMesh;

    std::vector<Qt3DCore::QEntity*> mMotionNodes;
    std::vector<Qt3DCore::QEntity*> mMotionRadiuses;
    std::vector<Qt3DCore::QEntity*> mMeshVertexes;
};

#endif