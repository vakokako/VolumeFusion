#include <af/VolumeFusion.h>
#include <af/Constants.h>
#include <af/ui/MainWindow.h>
#include <af/ui/ModifySignaler.h>
#include <af/ui/ModifySignalerQt.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkEventQtSlotConnect.h>
#include <vtkGlyph3D.h>
#include <vtkLine.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>
#include <vtkVertexGlyphFilter.h>

#include <QtConcurrent/QtConcurrent>
#include <mutex>

#include "af/ui/warpMeshTmp.h"
#include "vtkGenericOpenGLRenderWindow.h"

const std::string settingsPresetsPath = "/home/mvankovych/Uni/thesis/VolumeFusion/dataPresets.json";

void upperbodyPreset(af::Settings& settings) {
    // af::Settings settings;

    settings.dataFolder         = "/home/mvankovych/Uni/thesis/VolumeFusion/data/upperbody/data/01";
    settings.cameraCount        = 1;
    settings.depthFilesPattern  = "frame-******.depth.png";
    settings.framesRange.first  = 200;
    settings.framesRange.second = 400;
    settings.depthThreshold     = 1.2f;
    settings.tsdfSize           = Vec3f(1.0f, 1.0f, 1.0f);
    settings.tsdfDim            = Vec3i(400, 400, 400);
    settings.tsdfCenter         = Vec3f(-0.00199827f, 0.119444f, 0.8738f);
    settings.motionGraphRadius  = 0.03f;

    // return settings;
}

void multipersonPreset(af::Settings& settings) {
    // af::Settings settings;

    settings.dataFolder         = "/home/mvankovych/Uni/thesis/VolumeFusion/data/3_person_walking";
    settings.cameraCount        = 2;
    settings.depthFilesPattern  = "**********_depth.tiff";
    settings.framesRange.first  = 1;
    settings.framesRange.second = 1000;
    settings.depthThreshold     = 20.2f;
    settings.tsdfSize           = Vec3f(3.0f, 2.0f, 3.0f);
    settings.tsdfDim            = Vec3i(400, 400, 400);
    settings.tsdfCenter         = Vec3f(0.3f, -1.2f, 2.8f);
    settings.motionGraphRadius  = 0.1f;

    // return settings;
}

// Constructor
MainWindow::MainWindow()
    : _volumeFusionResult{_motionGraph,
                          _points["canon"],
                          _points["canonWarped"],
                          _points["graph"],
                          _points["frame"],
                          _points["meshCanon"],
                          _faces["meshCanon"],
                          _correspondenceCanonToFrame,
                          _graphKnns,
                          _tsdfCenter,
                          _volumeFusionStepper,
                          _volumeFusionRunning,
                          _dataLoadMutexDirectReconstr,
                          _objectMesh},
      _meshReconstructionBuffers{_points["frame"],     _points["meshCanon"],         _faces["meshCanon"],
                                 _volumeFusionStepper, _directReconstructionRunning, _dataLoadMutexDirectReconstr,
                                 _objectMesh},
      _pointCloudBuffers{_points["frame"], _volumeFusionStepper, _directReconstructionRunning, _dataLoadMutexDirectReconstr} {
    // set default state of running to true
    _volumeFusionRunning.test_and_set();
    _directReconstructionRunning.test_and_set();

    this->setupUi(this);

    loadSettingsPresets();

    this->qvtkWidget->setVisible(true);
    this->checkDisplayVisualization->setChecked(this->qvtkWidget->isVisible());

    _tsdfCenter.object() = Vec3f(0.f, 0.f, 0.f);

    this->sliderPointSize->setValue(6);

    _stepperWidget = new StepperWidget(_volumeFusionStepper);
    this->mainTabLayout->addWidget(_stepperWidget);

    vtkNew<vtkEventQtSlotConnect> slotConnector;
    this->_connections = slotConnector;

    addPointVertexComponent("canon", "Tomato", 6);
    addPointVertexComponent("canonWarped", "Royalblue", 6);
    addPointVertexComponent("frame", "Turquoise", 6);
    addMeshComponent("meshCanon", "White", 6);
    // addPointVertexComponent("meshCanon", "teal", 6);
    addLinesComponent("graph", "Magenta", 6);
    addLinesComponent("correspondence", "Magenta", 2);
    // addLinesComponent("correspondences", "Green", 6);
    buildVerticies(_points["graph"].points(), "graphVerticies", "Magenta", 16);
    buildSpheres(_points["graph"].points(), 0.03, "graphSpheres", "Magenta", 6);
    addBoundingBoxComponent("boundingbox", "Black", 6);

    // VTK Renderer
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    for (auto&& actor : _actors) {
        renderer->AddActor(actor.second);
    }
    vtkNew<vtkNamedColors> colors;
    renderer->GradientBackgroundOn();
    renderer->SetBackground(colors->GetColor3d("SlateBlue").GetData());
    renderer->SetBackground2(colors->GetColor3d("Black").GetData());

    auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    this->qvtkWidget->setRenderWindow(renderWindow);

    vtkSmartPointer<InteractorStyleSelect> style = vtkSmartPointer<InteractorStyleSelect>::New();
    style->Data                                  = _polydatas["frame"];
    style->DataActor                             = _actors["frame"];
    renderWindow->GetInteractor()->SetInteractorStyle(style);

    // setDisplayGraphRadiuses(false);
    // setDisplayMeshWarped(false);

    this->setupConnections();
    this->comboboxPresets->setCurrentIndex(0);
    if (!_settingsPresets.empty()) {
        _settings = _settingsPresets[0];
    }
    this->setSettings(_settings);
}

void MainWindow::updateVtkViewComponent() {
    // ModifySignalerQt* send = static_cast<ModifySignalerQt*>(sender());
    // for (auto&& points : _points) {
    //     if (points.second.signaler().qt() != send)
    //         continue;
    //     points.second.points()->
    // }
    // for (auto&& plyData : _polydatas) {
    //     plyData.second->Modified();
    // }
    // _polydatas[componentName]->Modified();

    this->qvtkWidget->renderWindow()->Render();
}

void MainWindow::graphModified() {
    vtkSmartPointer<vtkCellArray> lines = _cellArrays["graph"];
    lines->Reset();
    std::array<long long, 2> line;
    for (std::size_t i = 0; i < _graphKnns.vector().size(); ++i) {
        line[0] = i;
        for (int k = 0; k < _graphKnns.vector()[i].size(); ++k) {
            unsigned int knn = _graphKnns.vector()[i][k];
            if (knn >= _graphKnns.vector().size()) {
                continue;
            }
            line[1] = knn;
            lines->InsertNextCell(line.size(), line.data());
        }
    }
    // _points["graph"].points()->Modified();
    lines->Modified();
    // _polydatas["graph"]->Modified();
    updateVtkViewComponent();
}

void MainWindow::correspondenceModified() {
    auto canonPoints   = _points["canon"].points();
    auto framePoints   = _points["frame"].points();
    auto correspPoints = _points["correspondence"].points();
    auto canonSize     = canonPoints->GetNumberOfPoints();
    auto frameSize     = framePoints->GetNumberOfPoints();
    auto correspLines  = _cellArrays["correspondence"];
    {
        std::scoped_lock lock(_points["canon"].mutex(), _points["frame"].mutex(), _points["correspondence"].mutex());

        correspPoints->Resize(canonSize + frameSize);
        correspPoints->SetNumberOfPoints(canonSize + frameSize);

        std::copy((Vec3f*)canonPoints->GetVoidPointer(0), (Vec3f*)canonPoints->GetVoidPointer(0) + canonSize,
                  (Vec3f*)correspPoints->GetVoidPointer(0));
        std::copy((Vec3f*)framePoints->GetVoidPointer(0), (Vec3f*)framePoints->GetVoidPointer(0) + frameSize,
                  (Vec3f*)correspPoints->GetVoidPointer(0) + canonSize);

        correspLines->Reset();

        std::array<long long, 2> line;
        for (auto& corresp : _correspondenceCanonToFrame.vector()) {
            line[0] = corresp[0];
            line[1] = corresp[1] + canonSize;
            correspLines->InsertNextCell(line.size(), line.data());
        }
    }

    correspPoints->Modified();
    correspLines->Modified();
    updateVtkViewComponent();
}

void MainWindow::directMeshModified() {
    std::lock_guard<std::mutex> lock(_dataLoadMutexDirectReconstr);

    vtkSmartPointer<vtkCellArray>& triangles = _cellArrays["meshCanon"];
    af::BufferVector<Vec3i>& faces           = _faces["meshCanon"];
    triangles->Reset();
    std::array<long long, 3> triangle;
    for (auto& face : faces.vector()) {
        std::copy_n(face.begin(), face.size(), triangle.begin());
        triangles->InsertNextCell(triangle.size(), triangle.data());
    }
    _points["meshCanon"].points()->Modified();
    triangles->Modified();
    _polydatas["meshCanon"]->Modified();
    updateVtkViewComponent();
}

void MainWindow::startStopVolumeFusionDetached() {
    if (_threads["volumeFusion"].isRunning()) {
        _volumeFusionRunning.clear();
        return;
    }

    this->getSettings(this->_settings);
    _threads["volumeFusion"] = QtConcurrent::run(
        std::bind(&af::runVolumeFusion,
                  std::ref(_volumeFusionResult),  //   std::ref(this->_points["canon"]), std::ref(this->_points["frame"]),
                                                  //   std::ref(this->_points["graph"]),
                  std::ref(this->_settings)));
    _watchers["volumeFusion"].setFuture(_threads["volumeFusion"]);
    setStateRunning();
}

void MainWindow::startVolumeFusion() {
    if (_threads["volumeFusion"].isRunning())
        return;

    this->_stepperWidget->setEnabled(false);
    this->getSettings(this->_settings);

    af::runVolumeFusion(_volumeFusionResult, this->_settings);
}

void MainWindow::startMeshReconstructionStreamDetached() {
    if (_threads["meshReconstruction"].isRunning()) {
        _directReconstructionRunning.clear();
        return;
    }

    this->getSettings(this->_settings);
    _threads["meshReconstruction"] = QtConcurrent::run(
        std::bind(&af::startMeshReconstructionStream,
                  std::ref(_meshReconstructionBuffers),  //   std::ref(this->_points["canon"]), std::ref(this->_points["frame"]),
                                                         //   std::ref(this->_points["graph"]),
                  std::ref(this->_settings)));
    _watchers["meshReconstruction"].setFuture(_threads["meshReconstruction"]);
    setMeshReconstructionRunning();
}

void MainWindow::startPointCloudStreamDetached() {
    if (_threads["pointCloud"].isRunning()) {
        _directPointCloudRunning.clear();
        return;
    }

    this->getSettings(this->_settings);
    _threads["pointCloud"] = QtConcurrent::run(
        std::bind(&af::startPointCloudStream,
                  std::ref(_pointCloudBuffers),  //   std::ref(this->_points["canon"]), std::ref(this->_points["frame"]),
                                                 //   std::ref(this->_points["graph"]),
                  std::ref(this->_settings)));
    _watchers["pointCloud"].setFuture(_threads["pointCloud"]);
    setDirectPointCloudRunning();
}

void MainWindow::startMeshReconstructionStream() {
    if (_threads["meshReconstruction"].isRunning()) {
        return;
    }

    this->_stepperWidget->setEnabled(false);
    this->getSettings(this->_settings);

    af::startMeshReconstructionStream(_meshReconstructionBuffers, this->_settings);
    setMeshReconstructionRunning();
}

void MainWindow::setStateRunning() {
    this->pushStartVolume->setText("Stop Volume Fusion");
    // this->pushStartVolume->setEnabled(false);
}

void MainWindow::setStateAvailable() {
    this->pushStartVolume->setText("Start Volume Fusion");
    // this->pushStartVolume->setEnabled(true);
}
void MainWindow::setMeshReconstructionRunning() { this->pushStartMeshReconstruction->setText("Stop Direct Mesh Reconstruction"); }
void MainWindow::setMeshReconstructionAvailable() {
    this->pushStartMeshReconstruction->setText("Start Direct Mesh Reconstruction");
}
void MainWindow::setDirectPointCloudRunning() { this->pushStartPointCloudStream->setText("Stop Direct Point cloud Stream"); }
void MainWindow::setDirectPointCloudAvailable() { this->pushStartPointCloudStream->setText("Start Direct Point cloud Stream"); }

void MainWindow::setDisplayMesh(bool display) {
    _actors["canon"]->SetVisibility(display);
    this->qvtkWidget->renderWindow()->Render();
}
void MainWindow::setDisplayMeshWarped(bool display) {
    _actors["canonWarped"]->SetVisibility(display);
    this->qvtkWidget->renderWindow()->Render();
}
void MainWindow::setDisplayGraph(bool display) {
    _actors["graph"]->SetVisibility(display);
    _actors["graphVerticies"]->SetVisibility(display);
    this->qvtkWidget->renderWindow()->Render();
}
void MainWindow::setDisplayGraphRadiuses(bool display) {
    _actors["graphSpheres"]->SetVisibility(display);
    this->qvtkWidget->renderWindow()->Render();
}
void MainWindow::setDisplayFrameMesh(bool display) {
    _actors["frame"]->SetVisibility(display);
    this->qvtkWidget->renderWindow()->Render();
}

void MainWindow::setPointSize(int size) {
    _actors["canon"]->GetProperty()->SetPointSize(size);
    _actors["canonWarped"]->GetProperty()->SetPointSize(size);
    _actors["graph"]->GetProperty()->SetLineWidth(size);
    _actors["graphVerticies"]->GetProperty()->SetPointSize(size + 10);
    _actors["frame"]->GetProperty()->SetPointSize(size);
    this->qvtkWidget->renderWindow()->Render();
}

void MainWindow::setNormsExclStart(float value) { _settings.normExclAngleRange.first = value; }
void MainWindow::setNormsExclEnd(float value) { _settings.normExclAngleRange.second = value; }
void MainWindow::setGraphMinRadius(float value) {
    _spheres["graphSpheres"]->SetRadius(value);
    _polydatas["graphSpheres"]->Modified();
    this->qvtkWidget->renderWindow()->Render();
    _settings.motionGraphRadius = value;
}

void MainWindow::loadTsdfParamsFromUi() {
    _settings.tsdfSize     = spinboxDoubleVolSize->getValues();
    _settings.tsdfCenter   = spinboxDoubleVolCenter->getValues();
    _settings.tsdfRotation = spinboxVolRotation->getValues();
    // std::cout << "_tsdfCenter.object() : " << _tsdfCenter.object() << "\n";
    calculateBoundingBoxVerticies(_points["boundingbox"].points(), _settings.tsdfCenter, _settings.tsdfRotation,
                                  _settings.tsdfSize);
    _polydatas["boundingbox"]->Modified();
    this->qvtkWidget->renderWindow()->Render();
}

void MainWindow::tsdfGeometryChanged() {
    Vec3f& tsdfCenter = _tsdfCenter.object();
    calculateBoundingBoxVerticies(_points["boundingbox"].points(), tsdfCenter, _settings.tsdfRotation, _settings.tsdfSize);
    _polydatas["boundingbox"]->Modified();
    this->qvtkWidget->renderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetFocalPoint(
        tsdfCenter[0], tsdfCenter[1], tsdfCenter[2]);
    this->qvtkWidget->renderWindow()->Render();
}

void MainWindow::volumeFusionFinished() {
    setStateAvailable();
    // if (_objectMesh.savePly(_settings.dataFolder + "/testDump/objectMesh.ply"))
    //     std::cout << "object mesh .ply was saved.\n";
}

void MainWindow::addPointVertexComponent(const std::string& name, const std::string& color, int pointSize) {
    // if (_points.find(name) == _points.end())
    _points[name];

    buildVerticies(_points[name].points(), name, color, pointSize);
}

void MainWindow::buildVerticies(vtkSmartPointer<vtkPoints> points,
                                const std::string& name,
                                const std::string& color,
                                int pointSize) {
    if (_polydatas.find(name) == _polydatas.end())
        _polydatas[name] = vtkSmartPointer<vtkPolyData>::New();

    _polydatas[name]->SetPoints(points);

    vtkNew<vtkVertexGlyphFilter> vertexGlyph;
    vertexGlyph->AddInputData(_polydatas[name]);
    vertexGlyph->Update();

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(vertexGlyph->GetOutputPort());
    mapper->Update();

    _actors[name] = vtkSmartPointer<vtkActor>::New();
    _actors[name]->SetMapper(mapper);
    _actors[name]->GetProperty()->SetPointSize(pointSize);
    vtkNew<vtkNamedColors> colors;
    _actors[name]->GetProperty()->SetColor(colors->GetColor4d(color).GetData());
}

void MainWindow::buildSpheres(vtkSmartPointer<vtkPoints> points,
                              float radius,
                              const std::string& name,
                              const std::string& color,
                              int pointSize) {
    _spheres[name] = vtkSmartPointer<vtkSphereSource>::New();
    _spheres[name]->SetRadius(radius);
    _spheres[name]->SetPhiResolution(10);
    _spheres[name]->SetThetaResolution(10);

    _polydatas[name] = vtkSmartPointer<vtkPolyData>::New();
    _polydatas[name]->SetPoints(points);

    vtkSmartPointer<vtkGlyph3D> glyphs = vtkSmartPointer<vtkGlyph3D>::New();
    glyphs->AddInputData(_polydatas[name]);
    glyphs->SetSourceConnection(_spheres[name]->GetOutputPort());
    glyphs->ScalingOff();
    glyphs->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyphs->GetOutputPort());

    _actors[name] = vtkSmartPointer<vtkActor>::New();
    _actors[name]->SetMapper(mapper);
    _actors[name]->GetProperty()->SetOpacity(0.2);
    _actors[name]->GetProperty()->SetPointSize(pointSize);
    vtkNew<vtkNamedColors> colors;
    _actors[name]->GetProperty()->SetColor(colors->GetColor4d(color).GetData());
}

void MainWindow::addLinesComponent(const std::string& name, const std::string& color, int pointSize) {
    // if (_points.find(name) != _points.end())
    //     throw std::runtime_error("MainWindow::addLinesComponent(): " + name + " already exists.");

    _points[name];
    _cellArrays[name] = vtkSmartPointer<vtkCellArray>::New();

    _polydatas[name] = vtkSmartPointer<vtkPolyData>::New();
    _polydatas[name]->SetPoints(_points[name].points());
    _polydatas[name]->SetLines(_cellArrays[name]);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(_polydatas[name]);
    mapper->Update();

    _actors[name] = vtkSmartPointer<vtkActor>::New();
    _actors[name]->SetMapper(mapper);
    // _actors[name]->GetProperty()->SetPointSize(pointSize);
    _actors[name]->GetProperty()->SetLineWidth(pointSize);
    vtkNew<vtkNamedColors> colors;
    _actors[name]->GetProperty()->SetColor(colors->GetColor4d(color).GetData());
}

void MainWindow::addMeshComponent(const std::string& name, const std::string& color, int pointSize) {
    // if (_points.find(name) != _points.end())
    //     throw std::runtime_error("MainWindow::addLinesComponent(): " + name + " already exists.");

    _points[name].setSilent(true);
    _cellArrays[name] = vtkSmartPointer<vtkCellArray>::New();

    _polydatas[name] = vtkSmartPointer<vtkPolyData>::New();
    _polydatas[name]->SetPoints(_points[name].points());
    _polydatas[name]->SetPolys(_cellArrays[name]);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(_polydatas[name]);
    mapper->Update();

    _actors[name] = vtkSmartPointer<vtkActor>::New();
    _actors[name]->SetMapper(mapper);
    // _actors[name]->GetProperty()->SetPointSize(pointSize);
    _actors[name]->GetProperty()->SetLineWidth(pointSize);
    vtkNew<vtkNamedColors> colors;
    _actors[name]->GetProperty()->SetColor(colors->GetColor4d(color).GetData());
}

void MainWindow::addBoundingBoxComponent(const std::string& name, const std::string& color, int pointSize) {
    if (_points.find(name) != _points.end())
        throw std::runtime_error("MainWindow::addPointVertexComponent(): " + name + " already exists.");

    _points[name];

    calculateBoundingBoxVerticies(_points[name].points(), _tsdfCenter.object(), _settings.tsdfRotation, _settings.tsdfSize);

    std::vector<std::pair<int, int>> linesPoints;
    linesPoints.push_back({0, 1});
    linesPoints.push_back({0, 2});
    linesPoints.push_back({0, 4});
    linesPoints.push_back({1, 3});
    linesPoints.push_back({1, 5});
    linesPoints.push_back({2, 3});
    linesPoints.push_back({2, 6});
    linesPoints.push_back({3, 7});
    linesPoints.push_back({4, 5});
    linesPoints.push_back({4, 6});
    linesPoints.push_back({5, 7});
    linesPoints.push_back({6, 7});
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    for (std::size_t i = 0; i < linesPoints.size(); ++i) {
        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, linesPoints[i].first);
        line->GetPointIds()->SetId(1, linesPoints[i].second);
        lines->InsertNextCell(line);
    }

    _polydatas[name] = vtkSmartPointer<vtkPolyData>::New();
    _polydatas[name]->SetPoints(_points[name].points());
    _polydatas[name]->SetLines(lines);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputData(_polydatas[name]);
    mapper->Update();

    _actors[name] = vtkSmartPointer<vtkActor>::New();
    _actors[name]->SetMapper(mapper);
    _actors[name]->GetProperty()->SetPointSize(pointSize);
    vtkNew<vtkNamedColors> colors;
    _actors[name]->GetProperty()->SetColor(colors->GetColor4d(color).GetData());
    _actors[name]->GetProperty()->SetLineWidth(4);
}

void MainWindow::calculateBoundingBoxVerticies(vtkSmartPointer<vtkPoints> points,
                                               const Vec3f& center,
                                               const Vec3i& rotation,
                                               const Vec3f& size) {
    constexpr int verticiesCount = 8;
    points->Resize(verticiesCount);
    points->SetNumberOfPoints(verticiesCount);

    Vec3f halfSize = size / 2.f;
    std::vector<Vec3f> verticies(verticiesCount);

    Eigen::AngleAxisf xAngle(static_cast<float>(rotation(0)) / 180.f * M_PI, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf yAngle(static_cast<float>(rotation(1)) / 180.f * M_PI, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf zAngle(static_cast<float>(rotation(2)) / 180.f * M_PI, Eigen::Vector3f::UnitZ());

    Eigen::Quaternion<float> q   = xAngle * yAngle * zAngle;
    Eigen::Matrix3f tsdfRotation = q.matrix();

    // clang-format off
    verticies[0] = tsdfRotation * Vec3f(-halfSize[0], -halfSize[1], -halfSize[2]) + center;
    verticies[1] = tsdfRotation * Vec3f(-halfSize[0], -halfSize[1],  halfSize[2]) + center;
    verticies[2] = tsdfRotation * Vec3f(-halfSize[0],  halfSize[1], -halfSize[2]) + center;
    verticies[3] = tsdfRotation * Vec3f(-halfSize[0],  halfSize[1],  halfSize[2]) + center;
    verticies[4] = tsdfRotation * Vec3f( halfSize[0], -halfSize[1], -halfSize[2]) + center;
    verticies[5] = tsdfRotation * Vec3f( halfSize[0], -halfSize[1],  halfSize[2]) + center;
    verticies[6] = tsdfRotation * Vec3f( halfSize[0],  halfSize[1], -halfSize[2]) + center;
    verticies[7] = tsdfRotation * Vec3f( halfSize[0],  halfSize[1],  halfSize[2]) + center;
    // clang-format on
    for (int i = 0; i < points->GetNumberOfPoints(); ++i) {
        points->SetPoint(i, verticies[i][0], verticies[i][1], verticies[i][2]);
    }
}

Vec3f MainWindow::getVolSizeUi() {
    return Vec3f();
    // return spinboxDoubleVolSize->getValues();
}

void MainWindow::setVolSizeUi([[maybe_unused]] const Vec3f& volSize) {
    // spinboxDoubleVolSizeX->setValue(volSize[0]);
    // spinboxDoubleVolSizeY->setValue(volSize[1]);
    // spinboxDoubleVolSizeZ->setValue(volSize[2]);
}

void MainWindow::getSettings(af::Settings& settings) {
    settings.tsdfSize                  = this->spinboxDoubleVolSize->getValues();
    settings.tsdfDim                   = this->spinboxVolDim->getValues();
    settings.tsdfCenter                = this->spinboxDoubleVolCenter->getValues();
    settings.tsdfRotation              = this->spinboxVolRotation->getValues();
    settings.dataFolder                = this->lineEditData->text().toStdString();
    settings.cameraCount               = this->spinboxCamerasCount->value();
    settings.depthFilesPattern         = this->lineEditDepthFilesPattern->text().toStdString();
    settings.normExclAngleRange.first  = this->spinboxNormsExcl->_tuple[0]->value();
    settings.normExclAngleRange.second = this->spinboxNormsExcl->_tuple[1]->value();
    settings.framesRange.first         = this->spinboxFramesRange->at(0)->value();
    settings.framesRange.second        = this->spinboxFramesRange->at(1)->value();
    settings.frameBreak                = this->spinboxFrameBreak->value();
    settings.isMeshWarped              = this->checkIsMeshWarped->isChecked();
    settings.depthThreshold            = this->spinboxDepthFilter->value();
    settings.integrateTsdf             = this->checkIntegrateTsdf->isChecked();
    settings.bilateralFiltr            = this->checkBilateralFiltr->isChecked();
    settings.updateGraph               = this->checkUpdateGraph->isChecked();
    settings.useCholesky               = this->checkUseCholesky->isChecked();
    settings.useCopyHack               = this->checkUseCopyHack->isChecked();
    settings.bilateralD                = this->spinboxBilateralParams->at(0)->value();
    settings.bilateralSigmaI           = this->spinboxBilateralParams->at(1)->value();
    settings.bilateralSigmaS           = this->spinboxBilateralParams->at(2)->value();
    settings.bilateralThreshold        = this->spinboxBilateralParams->at(3)->value();
    settings.energyWeightDepth         = this->spinboxDoubleWeightDepth->value();
    settings.energyWeightMReg          = this->spinboxDoubleWeightMReg->value();
    settings.energyMinStep             = this->spinboxDoubleEnergyMinStep->value();
    settings.icpIterations             = this->spinboxIcpIterations->value();
    settings.motionGraphRadius         = this->spinboxDoubleGraphRadius->value();
    settings.motionGraphSampleDistance = this->spinboxDoubleGraphSampleDist->value();
    settings.correspThreshDist         = this->spinboxDoubleCorrespThreshDist->value();
    settings.tsdfDelta                 = this->spinboxDoubleTsdfDelta->value();
}

void MainWindow::setSettings(const af::Settings& settings) {
    auto silentWidgets = this->findChildren<QWidget*>();
    for (auto widget : silentWidgets) {
        widget->blockSignals(true);
    }
    this->spinboxDoubleVolSize->setValues(settings.tsdfSize);
    this->spinboxVolDim->setValues(settings.tsdfDim);
    this->spinboxDoubleVolCenter->setValues(settings.tsdfCenter);
    this->spinboxVolRotation->setValues(settings.tsdfRotation);
    this->lineEditData->setText(QString::fromStdString(settings.dataFolder));
    this->spinboxCamerasCount->setValue(settings.cameraCount);
    this->lineEditDepthFilesPattern->setText(QString::fromStdString(settings.depthFilesPattern));
    this->spinboxNormsExcl->setValues({settings.normExclAngleRange.first, settings.normExclAngleRange.second});
    this->spinboxFramesRange->setValues({settings.framesRange.first, settings.framesRange.second});
    this->spinboxFrameBreak->setValue(settings.frameBreak);
    this->checkIsMeshWarped->setChecked(settings.isMeshWarped);
    this->spinboxDepthFilter->setValue(settings.depthThreshold);
    this->checkIntegrateTsdf->setChecked(settings.integrateTsdf);
    this->checkBilateralFiltr->setChecked(settings.bilateralFiltr);
    this->checkUpdateGraph->setChecked(settings.updateGraph);
    this->checkUseCholesky->setChecked(settings.useCholesky);
    this->checkUseCopyHack->setChecked(settings.useCopyHack);
    this->spinboxBilateralParams->setValues(
        {(float)settings.bilateralD, settings.bilateralSigmaI, settings.bilateralSigmaS, settings.bilateralThreshold});
    this->spinboxDoubleWeightDepth->setValue(settings.energyWeightDepth);
    this->spinboxDoubleWeightMReg->setValue(settings.energyWeightMReg);
    this->spinboxDoubleEnergyMinStep->setValue(settings.energyMinStep);
    this->spinboxIcpIterations->setValue(settings.icpIterations);
    this->spinboxDoubleGraphRadius->setValue(settings.motionGraphRadius);
    this->spinboxDoubleGraphSampleDist->setValue(settings.motionGraphSampleDistance);
    this->spinboxDoubleCorrespThreshDist->setValue(settings.correspThreshDist);
    this->spinboxDoubleTsdfDelta->setValue(settings.tsdfDelta);

    for (auto widget : silentWidgets) {
        widget->blockSignals(false);
    }
    loadTsdfParamsFromUi();
}

void MainWindow::loadSettingsPresets() {
    _settingsPresets = af::loadSettingsPresets(settingsPresetsPath);

    comboboxPresets->clear();
    for (auto& preset : _settingsPresets) {
        comboboxPresets->addItem(QString::fromStdString(preset.dataFolder));
    }
}

af::Settings& MainWindow::currentSettingsPreset() { return _settingsPresets[comboboxPresets->currentIndex()]; }

void MainWindow::setupConnections() {
    // this->_connections->Connect(this->qvtkWidget->renderWindow()->GetInteractor(), vtkCommand::RightButtonPressEvent, this,
    //                            SLOT(slot_Rightclicked(vtkObject *, unsigned long, void *, void *)));
    connect(this->comboboxPresets, qOverload<int>(&QComboBox::currentIndexChanged), [&](int index) {
        this->_settings = this->_settingsPresets[index];
        this->setSettings(this->_settings);
    });
    connect(this->pushSaveSettings, &QPushButton::released, [&]() {
        this->_settingsPresets[this->comboboxPresets->currentIndex()] = this->_settings;
        af::saveSettingsPresets("/home/mvankovych/Uni/thesis/VolumeFusion/dataPresets.json", this->_settingsPresets);
    });

    connect(this->pushStartVolume, &QPushButton::released, this, &MainWindow::startStopVolumeFusionDetached);
    connect(this->pushTestVolume, &QPushButton::released, this, &MainWindow::startVolumeFusion);
    connect(this->pushStartMeshReconstruction, &QPushButton::released, this, &MainWindow::startMeshReconstructionStreamDetached);
    connect(this->pushStartPointCloudStream, &QPushButton::released, this, &MainWindow::startPointCloudStreamDetached);
    connect(this->pushTestDirect, &QPushButton::released, this, &MainWindow::startMeshReconstructionStream);
    connect(&this->_watchers["volumeFusion"], &QFutureWatcher<void>::finished, this, &MainWindow::volumeFusionFinished);
    connect(&this->_watchers["meshReconstruction"], &QFutureWatcher<void>::finished, this,
            &MainWindow::setMeshReconstructionAvailable);
    connect(&this->_watchers["pointCloud"], &QFutureWatcher<void>::finished, this, &MainWindow::setDirectPointCloudAvailable);

    connect(this->checkDisplayVisualization, &QCheckBox::stateChanged, this->qvtkWidget, &QVTKOpenGLNativeWidget::setVisible);
    connect(this->checkDisplayMeshCanon, &QCheckBox::stateChanged, [&](bool state) {
        _actors["meshCanon"]->SetVisibility(state);
        this->qvtkWidget->renderWindow()->Render();
    });
    connect(this->checkDisplayPointsCanon, &QCheckBox::stateChanged, this, &MainWindow::setDisplayMesh);
    connect(this->checkDisplayPointsWarped, &QCheckBox::stateChanged, this, &MainWindow::setDisplayMeshWarped);
    connect(this->checkDisplayGraph, &QCheckBox::stateChanged, this, &MainWindow::setDisplayGraph);
    connect(this->checkDisplayGraphRadiuses, &QCheckBox::stateChanged, this, &MainWindow::setDisplayGraphRadiuses);
    connect(this->checkDisplayPointsFrame, &QCheckBox::stateChanged, this, &MainWindow::setDisplayFrameMesh);
    connect(this->checkDisplayTsdfBox, &QCheckBox::stateChanged, [this](bool checked) {
        _actors["boundingbox"]->SetVisibility(checked);
        this->qvtkWidget->renderWindow()->Render();
    });
    connect(this->checkDisplayCorrespondences, &QCheckBox::stateChanged, [&](bool checked) {
        _actors["correspondence"]->SetVisibility(checked);
        this->qvtkWidget->renderWindow()->Render();
    });
    connect(this->sliderPointSize, &QSlider::valueChanged, this, &MainWindow::setPointSize);

    connect(this->spinboxNormsExcl->_tuple[0], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::setNormsExclStart);
    connect(this->spinboxNormsExcl->_tuple[1], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::setNormsExclEnd);
    connect(this->spinboxDoubleGraphRadius, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::setGraphMinRadius);
    connect(this->spinboxDoubleVolSize->_tuple[0], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxDoubleVolSize->_tuple[1], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxDoubleVolSize->_tuple[2], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxDoubleVolCenter->_tuple[0], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxDoubleVolCenter->_tuple[1], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxDoubleVolCenter->_tuple[2], qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxVolRotation->_tuple[0], qOverload<int>(&QSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxVolRotation->_tuple[1], qOverload<int>(&QSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);
    connect(this->spinboxVolRotation->_tuple[2], qOverload<int>(&QSpinBox::valueChanged), this,
            &MainWindow::loadTsdfParamsFromUi);

    for (auto&& points : _points) {
        connect(points.second.signaler().qt(), &ModifySignalerQt::modifiedSignal, this, &MainWindow::updateVtkViewComponent);
        // [&]() { this->updateVtkViewComponent(points.first); });
    }
    connect(_graphKnns.signaler().qt(), &ModifySignalerQt::modifiedSignal, this, &MainWindow::graphModified);
    connect(_tsdfCenter.signaler().qt(), &ModifySignalerQt::modifiedSignal, this, &MainWindow::tsdfGeometryChanged);
    connect(_correspondenceCanonToFrame.signaler().qt(), &ModifySignalerQt::modifiedSignal, this,
            &MainWindow::correspondenceModified);

    _points["meshCanon"].signaler().qt()->blockSignals(true);
    _points["frame"].signaler().qt()->blockSignals(true);
    connect(_faces["meshCanon"].signaler().qt(), &ModifySignalerQt::modifiedSignal, this, &MainWindow::directMeshModified);

    connect(spinboxFrameBreak, qOverload<int>(&QSpinBox::valueChanged), [&](int value) { this->_settings.frameBreak = value; });

    connect(pushWarpMesh, &QPushButton::released, [&]() {
        af::warpMesh(this->_points["meshCanon"], this->_points["meshCanon"], _motionGraph);
        directMeshModified();
    });
    connect(this->comboboxMeshType, qOverload<int>(&QComboBox::currentIndexChanged), [&](int index) {
        switch (index) {
            case 0:
                lineEditMeshName->setText("/canonical/frame-000000.canonical.vako");
                break;
            case 1:
                lineEditMeshName->setText("/reconstruction/frame-000000.mesh.vako");
                break;
            default:
                break;
        }
    });
    connect(pushSaveMesh, &QPushButton::released, [&]() {
        const std::string meshFilePath = _settings.dataFolder + "/../../" + lineEditMeshName->text().toStdString() + ".ply";
        af::savePlyTmp(meshFilePath, this->_points["meshCanon"], _faces["meshCanon"].vector());
    });
}