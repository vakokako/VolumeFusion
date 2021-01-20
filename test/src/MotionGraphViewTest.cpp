#include <gtest/gtest.h>

#include <QtGui/QGuiApplication>
#include <QtGui/QScreen>
#include <QtWidgets/QApplication>
#include <Qt3DExtras/qt3dwindow.h>
#include <Qt3DExtras/qforwardrenderer.h>
#include <QtWidgets/QWidget>
#include <QtWidgets/QHBoxLayout>
#include <Qt3DInput/QInputAspect>
#include <Qt3DRender/qcamera.h>
#include <Qt3DCore/qentity.h>
#include <Qt3DRender/qcameralens.h>
#include <Qt3DRender/qpointlight.h>
#include <Qt3DCore/qtransform.h>
#include <Qt3DExtras/QOrbitCameraController>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QTextEdit>

#include <af/qt/SceneController.h>
#include <af/Constants.h>
#include <af/GraphBuilder.h>
#include <chrono>

void build3dView(QWidget*& container, Qt3DCore::QEntity*& rootEntity) {
    Qt3DExtras::Qt3DWindow *view = new Qt3DExtras::Qt3DWindow();
    view->defaultFrameGraph()->setClearColor(QColor(QRgb(0x4d4d4f)));
    container = QWidget::createWindowContainer(view);
    QSize screenSize = view->screen()->size();
    container->setMinimumSize(QSize(200, 100));
    container->setMaximumSize(screenSize);

    Qt3DInput::QInputAspect *input = new Qt3DInput::QInputAspect;
    view->registerAspect(input);

    rootEntity = new Qt3DCore::QEntity();
    view->setRootEntity(rootEntity);

    Qt3DRender::QCamera *cameraEntity = view->camera();

    cameraEntity->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
    cameraEntity->setPosition(QVector3D(0, 0, 3.0f));
    cameraEntity->setUpVector(QVector3D(0, 1, 0));
    cameraEntity->setViewCenter(QVector3D(0, 0, 0));

    Qt3DCore::QEntity *lightEntity = new Qt3DCore::QEntity(rootEntity);
    Qt3DRender::QPointLight *light = new Qt3DRender::QPointLight(lightEntity);
    light->setColor("white");
    light->setIntensity(1);
    lightEntity->addComponent(light);
    Qt3DCore::QTransform *lightTransform = new Qt3DCore::QTransform(lightEntity);
    lightTransform->setTranslation(cameraEntity->position());
    lightEntity->addComponent(lightTransform);

    Qt3DExtras::QOrbitCameraController *camController = new Qt3DExtras::QOrbitCameraController(rootEntity);
    camController->setCamera(cameraEntity);
    camController->setLinearSpeed(camController->linearSpeed());
    camController->setLookSpeed(camController->lookSpeed());
}

QWidget* createMenu1(SceneController3d& vSceneController, MotionGraph& graph, Mesh& mesh) {
    QVBoxLayout *vLayout = new QVBoxLayout();
    vLayout->setAlignment(Qt::AlignTop);

    vLayout->addWidget(new QLabel("Position"));

    QHBoxLayout *hLayoutPosition = new QHBoxLayout();
    QDoubleSpinBox* inputPosX = new QDoubleSpinBox();
    QDoubleSpinBox* inputPosY = new QDoubleSpinBox();
    QDoubleSpinBox* inputPosZ = new QDoubleSpinBox();
    hLayoutPosition->addWidget(inputPosX);
    hLayoutPosition->addWidget(inputPosY);
    hLayoutPosition->addWidget(inputPosZ);

    vLayout->addLayout(hLayoutPosition);

    QHBoxLayout *hLayoutTransformR = new QHBoxLayout();
    QHBoxLayout *hLayoutTransformT = new QHBoxLayout();
    QDoubleSpinBox* inputTransformRX = new QDoubleSpinBox();
    QDoubleSpinBox* inputTransformRY = new QDoubleSpinBox();
    QDoubleSpinBox* inputTransformRZ = new QDoubleSpinBox();
    QDoubleSpinBox* inputTransformTX = new QDoubleSpinBox();
    QDoubleSpinBox* inputTransformTY = new QDoubleSpinBox();
    QDoubleSpinBox* inputTransformTZ = new QDoubleSpinBox();
    inputTransformRX->setValue(1.);
    inputTransformRY->setValue(1.);
    inputTransformRZ->setValue(1.);
    hLayoutTransformR->addWidget(inputTransformRX);
    hLayoutTransformR->addWidget(inputTransformRY);
    hLayoutTransformR->addWidget(inputTransformRZ);
    hLayoutTransformT->addWidget(inputTransformTX);
    hLayoutTransformT->addWidget(inputTransformTY);
    hLayoutTransformT->addWidget(inputTransformTZ);

    vLayout->addWidget(new QLabel("Transform R"));
    vLayout->addLayout(hLayoutTransformR);
    vLayout->addWidget(new QLabel("Transform t"));
    vLayout->addLayout(hLayoutTransformT);


    QHBoxLayout *hLayoutRadius = new QHBoxLayout();
    QDoubleSpinBox* inputRadius = new QDoubleSpinBox();
    hLayoutRadius->addWidget(inputRadius);

    vLayout->addWidget(new QLabel("Radius"));
    vLayout->addLayout(hLayoutRadius);

    QHBoxLayout *hLayoutActions = new QHBoxLayout();
    QPushButton* pushButtonAddGraphVertex = new QPushButton("Add graph vertex");
    QPushButton* pushButtonAddMeshVertex = new QPushButton("Add mesh vertex");
    hLayoutActions->addWidget(pushButtonAddGraphVertex);
    hLayoutActions->addWidget(pushButtonAddMeshVertex);
    vLayout->addLayout(hLayoutActions);

    QPushButton* pushButtonWarp = new QPushButton("Warp");
    vLayout->addWidget(pushButtonWarp);
    QPushButton* pushButtonUpdateGraph = new QPushButton("UpdateGraph");
    vLayout->addWidget(pushButtonUpdateGraph);

    QTextEdit* textEditLog = new QTextEdit();
    QPushButton* pushButtonRefreshLog = new QPushButton("Refresh log");
    vLayout->addWidget(textEditLog);
    vLayout->addWidget(pushButtonRefreshLog);
    QObject::connect(pushButtonRefreshLog, &QPushButton::released, [=, &vSceneController, &graph, &mesh]() {
        textEditLog->setText(textEditLog->toPlainText() + "\ngraph nodes count : " + QString::number(graph.graph().size()));
        textEditLog->setText(textEditLog->toPlainText() + "\nmesh vertices count : " + QString::number(mesh.vertexCloud().size()));
    });

    QObject::connect(pushButtonAddGraphVertex, &QPushButton::released, [=, &vSceneController, &graph]() {
        Vec3f pos;
        Mat4f transform = Mat4f::Identity();
        float radius = 0;
        pos[0] = inputPosX->value();
        pos[1] = inputPosY->value();
        pos[2] = inputPosZ->value();
        transform(0, 0) = inputTransformRX->value();
        transform(1, 1) = inputTransformRY->value();
        transform(2, 2) = inputTransformRZ->value();
        transform(0, 3) = inputTransformTX->value();
        transform(1, 3) = inputTransformTY->value();
        transform(2, 3) = inputTransformTZ->value();
        radius = inputRadius->value();
        graph.push_back(pos, radius, transform);
        vSceneController.updateView();
    });
    QObject::connect(pushButtonAddMeshVertex, &QPushButton::released, [=, &vSceneController, &mesh]() {
        Vec3f pos;
        pos[0] = inputPosX->value();
        pos[1] = inputPosY->value();
        pos[2] = inputPosZ->value();
        mesh.addVertex(pos, Vec3b());
        vSceneController.updateView();
    });
    QObject::connect(pushButtonWarp, &QPushButton::released, [=, &vSceneController, &graph, &mesh]() {
        graph.warp(mesh.vertexCloud().vec_, mesh.vertexCloud().vec_, Constants::motionGraphKNN);
        vSceneController.updateView();
    });
    QObject::connect(pushButtonUpdateGraph, &QPushButton::released, [=, &vSceneController, &graph, &mesh]() {
        textEditLog->setText(textEditLog->toPlainText() + "\ngraph build started...");
        auto start = std::chrono::steady_clock::now();
        af::buildGraph(graph, mesh.vertexCloud().vec_, Constants::motionGraphKNN, Constants::motionGraphRadius);
        auto end = std::chrono::steady_clock::now();
        textEditLog->setText(textEditLog->toPlainText() + "\ngraph build finished!");
        textEditLog->setText(textEditLog->toPlainText() + "\ngraph nodes count : " + QString::number(graph.graph().size()));
        textEditLog->setText(textEditLog->toPlainText() + "\ngraph build time : " + QString::number(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + "ms");
        vSceneController.updateView();
    });


    QWidget* tabMenu = new QWidget();
    tabMenu->setLayout(vLayout);

    return tabMenu;
}

QWidget* createSettingsMenu(SceneController3d& vSceneController, MotionGraph& graph, Mesh& mesh) {
    QVBoxLayout *vLayout = new QVBoxLayout();
    vLayout->setAlignment(Qt::AlignTop);

    std::vector<QCheckBox*> settings;
    settings.push_back(new QCheckBox("motion nodes visible"));
    settings.push_back(new QCheckBox("motion nodes radiuses visible"));
    settings.push_back(new QCheckBox("mesh vertices visible"));
    for (auto &&checkBox : settings) {
        checkBox->setChecked(true);
    }

    QObject::connect(settings[0], &QCheckBox::stateChanged, [=, &vSceneController](bool isVisible) {
        vSceneController.setMotionNodesVisible(isVisible);
    });
    QObject::connect(settings[1], &QCheckBox::stateChanged, [=, &vSceneController](bool isVisible) {
        vSceneController.setMotionRadiusesVisible(isVisible);
    });
    QObject::connect(settings[2], &QCheckBox::stateChanged, [=, &vSceneController](bool isVisible) {
        vSceneController.setMeshVerticesVisible(isVisible);
    });

    for (auto &&checkBox : settings) {
        vLayout->addWidget(checkBox);
    }

    QWidget* tabMenu = new QWidget();
    tabMenu->setLayout(vLayout);

    return tabMenu;
}

void buildWindow(SceneController3d& vSceneController, MotionGraph& graph, Mesh& mesh) {
    QWidget* container;
    Qt3DCore::QEntity* rootEntity;
    build3dView(container, rootEntity);
    vSceneController.setRootEntity(rootEntity);

    QWidget *widget = new QWidget;
    QHBoxLayout *hLayout = new QHBoxLayout(widget);
    QVBoxLayout *vLayout = new QVBoxLayout();
    vLayout->setAlignment(Qt::AlignTop);
    hLayout->addWidget(container, 1);
    hLayout->addLayout(vLayout);

    QTabWidget* tabWidget = new QTabWidget;
    tabWidget->addTab(createMenu1(vSceneController, graph, mesh), "Menu1");
    tabWidget->addTab(createSettingsMenu(vSceneController, graph, mesh), "Settings");

    vLayout->addWidget(tabWidget);

    widget->setWindowTitle(QStringLiteral("Basic shapes"));
    widget->show();
    widget->resize(3000, 2000);
}

#include <af/dataset.h>
#include <af/CameraModel.h>
#include <af/TSDFVolume.h>
#include <af/VertexManipulation.h>
#include <af/MarchingCubes.h>
#include <fstream>
void compute(Mesh& outputMesh) {


    Vec3i volDim(256, 256, 256);
    Vec3f volSize(1.5f, 1.5f, 1.5f);

    // load camera intrinsics
    af::loadIntrinsics(Constants::dataFolder + "/depthIntrinsics.txt", CameraModel::KDepth);
    std::cout << "K depth: " << std::endl
              << CameraModel::KDepth << std::endl;

    float delta = 0.02f;

    Mat4f poseVolume = Mat4f::Identity();
    cv::Mat color, depth;


    TSDFVolume* tsdfResult = new TSDFVolume(volDim, volSize, CameraModel::KDepth);
    tsdfResult->setDelta(delta);

    // load input frame
    af::loadFrame(Constants::dataFolder, 0, color, depth);
    af::filterDepth(depth, Constants::depthThreshold);

    cv::Mat vertexMap;
    af::depthToVertexMap(CameraModel::KDepth, depth, vertexMap);
    Vec3f frameCentroid = af::centroid(vertexMap);
    poseVolume.topRightCorner<3, 1>() = frameCentroid;


    if (std::ifstream(Constants::appFolder + "tsdfResult.txt").good()) {
        tsdfResult->load(Constants::appFolder + "tsdfResult.txt");
    } else {
        tsdfResult->integrate(poseVolume, color, depth);
        tsdfResult->save(Constants::appFolder + "tsdfResult.txt");
    }

    bool tsdfvluesExist = false;
    size_t count = 0;
    for (size_t i = 0; i < tsdfResult->tsdf().size(); ++i) {
        if (tsdfResult->tsdf()[i] > -1.f && tsdfResult->tsdf()[i] < 1.f) {
            tsdfvluesExist = true;
            ++count;
        }
    }
    EXPECT_TRUE(tsdfvluesExist);
    EXPECT_TRUE(count > 50);


    MarchingCubes mc(volDim, volSize);
    mc.computeIsoSurface(outputMesh, tsdfResult->tsdf(), tsdfResult->tsdfWeights(), tsdfResult->colorR(), tsdfResult->colorG(), tsdfResult->colorB());

}

TEST(MotionGraphViewTest, QtRender) {
    int argc = 1;
    char* argv = "app.exe";
    QApplication app(argc, &argv);


    // MotionGraph graph1;
    // Mesh mesh1;
    // SceneController3d vSceneController;
    // buildWindow(vSceneController, graph1, mesh1);
    // vSceneController.setMotionGraph(graph);
    // vSceneController.setMesh(mesh);

    // const std::size_t count = 8;
    // const float size = 1.5f;
    // const float start = -(size / 2.);
    // const float step = size / count;
    // for (std::size_t i = 0; i < count; ++i) {
    //     for (std::size_t j = 0; j < count; ++j) {
    //         for (std::size_t l = 0; l < count; ++l) {
    //             mesh.addVertex(Vec3f(start + i * step, start + j * step, start + l * step), Vec3b());
    //         }
    //     }
    // }
    // compute(mesh);
    MotionGraph graph;
    Mesh mesh;
    mesh.loadPly(Constants::dataFolder + "/testDump/mesh3_without_mask.ply");
    float radius = Constants::motionGraphRadius * 0.1;
    std::size_t graphSize = graph.graph().size();
    std::cout << "graph.graph().size() : \n" << graph.graph().size() << "\n";
    af::buildGraph(graph, mesh.vertexCloud().vec_, Constants::motionGraphKNN, radius);
    std::cout << "radius : \n" << radius << "\n";
    std::cout << "Settings::motionGraphKNN : \n" << Constants::motionGraphKNN << "\n";
    std::cout << "mesh.vertexCloud().size() : " << mesh.vertexCloud().size() << "\n";
    std::cout << "graph.graph().size() : \n" << graph.graph().size() << "\n";
    // mesh.clear();
    // vSceneController.updateView();

    // vSceneController.addCustom(Vec3f(0, 0, 0), 1.f, Qt::blue, Settings::dataFolder + "/testDump/mesh3_without_mask.ply");
    // vSceneController.addCustom(Vec3f(0, 0, 0), 1.f, Qt::blue, Settings::dataFolder + "/../reconstruction/frame-000000.mesh.ply");
    // vSceneController.addCustom(Vec3f(0, 0, 0), 0.05f, Qt::blue, Settings::dataFolder + "/toyplane.obj");


    // app.exec();
}