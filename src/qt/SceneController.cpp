#include <af/qt/SceneController.h>

#include <QtCore/QDebug>
#include <iostream>

SceneController3d::SceneController3d(Qt3DCore::QEntity* rootEntity)
    : mRootEntity(rootEntity), mMotionGraph(nullptr), mMesh(nullptr) {}

SceneController3d::~SceneController3d() {}

void SceneController3d::setRootEntity(Qt3DCore::QEntity* rootEntity) { mRootEntity = rootEntity; }

void SceneController3d::setMotionGraph(const MotionGraph& cGraph) {
    mMotionGraph = &cGraph;
    updateView();
}

void SceneController3d::setMesh(const Mesh& cMesh) {
    mMesh = &cMesh;
    updateView();
}

void SceneController3d::addMotionVertex(const Vec3f& cPos, const float cRadius) {
    mMotionNodes.push_back(addCube(cPos, 0.006f, Qt::red));
    mMotionRadiuses.push_back(addSphere(cPos, cRadius, Qt::red, 0.4f, 10));
}

void SceneController3d::addMeshVertex(const Vec3f& cPos) { mMeshVertexes.push_back(addCube(cPos, 0.003f, Qt::blue)); }

bool SceneController3d::isAllComponentsSet() {
    if (mMesh == nullptr)
        return false;
    if (mMotionGraph == nullptr)
        return false;
    return true;
}

void SceneController3d::updateView() {
    if (!isAllComponentsSet())
        return;

    std::size_t graphUpdateEnd = mMotionNodes.size();
    std::size_t meshUpdateEnd  = mMeshVertexes.size();

    if (mMotionGraph->graph().size() > mMotionNodes.size()) {
        graphUpdateEnd = mMotionNodes.size();
        for (std::size_t i = mMotionNodes.size(); i < mMotionGraph->graph().size(); ++i) {
            addMotionVertex(mMotionGraph->graph().vec_[i], mMotionGraph->radiuses()[i]);
        }
    } else if (mMotionGraph->graph().size() < mMotionNodes.size()) {
        for (std::size_t i = mMotionNodes.size() - 1; i >= mMotionGraph->graph().size(); --i) {
            remove(mMotionNodes[i]);
            remove(mMotionRadiuses[i]);
        }
        mMotionNodes.resize(mMotionGraph->graph().size());
        mMotionRadiuses.resize(mMotionGraph->graph().size());
        graphUpdateEnd = mMotionNodes.size();
    }

    if (mMesh->vertexCloud().size() > mMeshVertexes.size()) {
        meshUpdateEnd = mMeshVertexes.size();
        for (std::size_t i = mMeshVertexes.size(); i < mMesh->vertexCloud().size(); ++i) {
            addMeshVertex(mMesh->vertexCloud().vec_[i]);
        }
    } else if (mMesh->vertexCloud().size() < mMeshVertexes.size()) {
        for (std::size_t i = mMeshVertexes.size() - 1; i >= mMesh->vertexCloud().size(); --i) {
            remove(mMeshVertexes[i]);
        }
        mMeshVertexes.resize(mMesh->vertexCloud().size());
        meshUpdateEnd = mMeshVertexes.size();
    }

    for (std::size_t i = 0; i < graphUpdateEnd; ++i) {
        setPos(mMotionNodes[i], mMotionGraph->graph().vec_[i]);
        setPos(mMotionRadiuses[i], mMotionGraph->graph().vec_[i]);
        setRadius(mMotionRadiuses[i], mMotionGraph->radiuses()[i]);
    }
    for (std::size_t i = 0; i < meshUpdateEnd; ++i) {
        setPos(mMeshVertexes[i], mMesh->vertexCloud().vec_[i]);
    }
}

void SceneController3d::setMotionNodesVisible(const bool cIsVisible) {
    for (auto&& node : mMotionNodes) {
        node->setEnabled(cIsVisible);
    }
}
void SceneController3d::setMotionRadiusesVisible(const bool cIsVisible) {
    for (auto&& node : mMotionRadiuses) {
        node->setEnabled(cIsVisible);
    }
}
void SceneController3d::setMeshVerticesVisible(const bool cIsVisible) {
    for (auto&& node : mMeshVertexes) {
        node->setEnabled(cIsVisible);
    }
}

Qt3DCore::QEntity* SceneController3d::addSphere(const Vec3f& cPos,
                                                const float cRadius,
                                                const QColor& cColor,
                                                const float cAlpha,
                                                const int cSlices) {
    if (mRootEntity == nullptr) {
        throw std::runtime_error("SceneController3d::addSphere(): root entity cannot be nullptr.");
    }
    // Sphere shape data
    Qt3DExtras::QSphereMesh* sphereMesh = new Qt3DExtras::QSphereMesh();
    sphereMesh->setRings(cSlices);
    sphereMesh->setSlices(cSlices);
    sphereMesh->setRadius(cRadius);

    // Sphere mesh transform
    Qt3DCore::QTransform* sphereTransform = new Qt3DCore::QTransform();

    sphereTransform->setTranslation(QVector3D(cPos[0], cPos[1], cPos[2]));

    Qt3DExtras::QPhongAlphaMaterial* sphereMaterial = new Qt3DExtras::QPhongAlphaMaterial();
    sphereMaterial->setDiffuse(cColor);
    sphereMaterial->setAmbient(cColor);
    sphereMaterial->setAlpha(cAlpha);
    // QColor(QRgb(0xa69929))

    // Sphere
    Qt3DCore::QEntity* sphereEntity = new Qt3DCore::QEntity(mRootEntity);
    sphereEntity->addComponent(sphereMesh);
    sphereEntity->addComponent(sphereMaterial);
    sphereEntity->addComponent(sphereTransform);

    return sphereEntity;
}

Qt3DCore::QEntity* SceneController3d::addCube(const Vec3f& cPos,
                                              const float cRadius,
                                              const QColor& cColor,
                                              const float cAlpha,
                                              const int cSlices) {
    if (mRootEntity == nullptr) {
        throw std::runtime_error("SceneController3d::addSphere(): root entity cannot be nullptr.");
    }
    // Cuboid shape data
    Qt3DExtras::QCuboidMesh* cuboid = new Qt3DExtras::QCuboidMesh();

    // CuboidMesh Transform
    Qt3DCore::QTransform* cuboidTransform = new Qt3DCore::QTransform();
    cuboidTransform->setScale(cRadius);
    cuboidTransform->setTranslation(QVector3D(cPos[0], cPos[1], cPos[2]));

    Qt3DExtras::QPhongAlphaMaterial* cuboidMaterial = new Qt3DExtras::QPhongAlphaMaterial();
    cuboidMaterial->setAlpha(cAlpha);
    cuboidMaterial->setDiffuse(cColor);
    cuboidMaterial->setAmbient(cColor);

    // Cuboid
    Qt3DCore::QEntity* cuboidEntity = new Qt3DCore::QEntity(mRootEntity);
    cuboidEntity->addComponent(cuboid);
    cuboidEntity->addComponent(cuboidMaterial);
    cuboidEntity->addComponent(cuboidTransform);

    return cuboidEntity;
}

Qt3DCore::QEntity* SceneController3d::addCustom(const Vec3f& cPos,
                                              const float cRadius,
                                              const QColor& cColor,
                                              const std::string& cFilePath) {
    if (mRootEntity == nullptr) {
        throw std::runtime_error("SceneController3d::addSphere(): root entity cannot be nullptr.");
    }
    // Cuboid shape data
    Qt3DRender::QMesh* mesh = new Qt3DRender::QMesh();
    mesh->setSource(QUrl::fromLocalFile(QString::fromStdString(cFilePath)));

    // CuboidMesh Transform
    Qt3DCore::QTransform* cuboidTransform = new Qt3DCore::QTransform();
    cuboidTransform->setScale(cRadius);
    cuboidTransform->setTranslation(QVector3D(cPos[0], cPos[1], cPos[2]));

    Qt3DExtras::QPhongAlphaMaterial* cuboidMaterial = new Qt3DExtras::QPhongAlphaMaterial();
    cuboidMaterial->setAlpha(1.f);
    cuboidMaterial->setDiffuse(cColor);
    cuboidMaterial->setAmbient(cColor);

    // Cuboid
    Qt3DCore::QEntity* cuboidEntity = new Qt3DCore::QEntity(mRootEntity);
    cuboidEntity->addComponent(mesh);
    cuboidEntity->addComponent(cuboidMaterial);
    cuboidEntity->addComponent(cuboidTransform);

    return cuboidEntity;
}

void SceneController3d::setPos(Qt3DCore::QEntity* entity, const Vec3f& cPos) {
    for (auto&& component : entity->components()) {
        Qt3DCore::QTransform* entityTransform = dynamic_cast<Qt3DCore::QTransform*>(component);
        if (!entityTransform)
            continue;
        entityTransform->setTranslation(QVector3D(cPos[0], cPos[1], cPos[2]));
        break;
    }
}

void SceneController3d::setRadius(Qt3DCore::QEntity* entity, const float cRadius) {
    for (auto&& component : entity->components()) {
        Qt3DExtras::QSphereMesh* sphereMesh = dynamic_cast<Qt3DExtras::QSphereMesh*>(component);
        if (!sphereMesh)
            continue;
        sphereMesh->setRadius(cRadius);
        break;
    }
}

void SceneController3d::clear(std::vector<Qt3DCore::QEntity*>& entities) {
    for (auto&& entity : entities) {
        remove(entity);
    }
    entities.clear();
}

void SceneController3d::remove(Qt3DCore::QEntity* entity) {
    for (auto&& component : entity->components()) {
        entity->removeComponent(component);
    }

    // entity->removeAllComponents();
    entity->setParent((Qt3DCore::QNode*)nullptr);
    entity->deleteLater();
    // delete entity;
}