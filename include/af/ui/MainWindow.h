#ifndef MAINWINDOW_CUH
#define MAINWINDOW_CUH

#include <af/Settings.h>
#include <af/Mesh.h>
#include <af/Constants.h>
#include <af/MotionGraph.h>
#include <af/eigen_extension.h>
#include <af/ui/MainWindowUI.h>
#include <af/ui/PointsQWrap.h>
#include <af/ui/VectorQWrap.h>
#include <vtkSmartPointer.h>

#include <QtCore/QFuture>
#include <QtCore/QFutureWatcher>
#include <QtWidgets/QMainWindow>
#include <map>

#include "af/VolumeFusion.h"
#include "af/MeshRecontructionStream.cuh"
#include "af/PointCloudStream.cuh"
#include "af/ui/BufferObject.h"
#include "af/ui/BufferVector.h"
#include "af/ui/BufferVtkPoints.h"

class vtkEventQtSlotConnect;
class vtkPolyData;
class vtkRenderer;
class vtkGenericOpenGLRenderWindow;
class vtkActor;
class vtkCellArray;
class vtkSphereSource;

class MainWindow : public QMainWindow, public MainWindowUI {
    Q_OBJECT
public:
    MainWindow();

public slots:

    void updateVtkViewComponent();
    void graphModified();
    void correspondenceModified();
    void directMeshModified();
    void startStopVolumeFusionDetached();
    void startVolumeFusion();
    void startMeshReconstructionStreamDetached();
    void startPointCloudStreamDetached();
    void startMeshReconstructionStream();
    void setStateRunning();
    void setStateAvailable();
    void setMeshReconstructionRunning();
    void setMeshReconstructionAvailable();
    void setDirectPointCloudRunning();
    void setDirectPointCloudAvailable();
    void setDisplayMesh(bool display);
    void setDisplayMeshWarped(bool display);
    void setDisplayGraph(bool display);
    void setDisplayGraphRadiuses(bool display);
    void setDisplayFrameMesh(bool display);
    void setPointSize(int size);
    void setNormsExclStart(float value);
    void setNormsExclEnd(float value);
    void setGraphMinRadius(float value);
    void loadTsdfParamsFromUi();
    void tsdfGeometryChanged();

    void volumeFusionFinished();

protected:
    void addPointVertexComponent(const std::string& name, const std::string& color, int pointSize = 3);
    void buildVerticies(vtkSmartPointer<vtkPoints> points, const std::string& name, const std::string& color, int pointSize);
    void buildSpheres(vtkSmartPointer<vtkPoints> points,
                      float radius,
                      const std::string& name,
                      const std::string& color,
                      int pointSize);
    void addLinesComponent(const std::string& name, const std::string& color, int pointSize = 3);
    void addMeshComponent(const std::string& name, const std::string& color, int pointSize = 3);
    void addBoundingBoxComponent(const std::string& name, const std::string& color, int pointSize = 3);
    void calculateBoundingBoxVerticies(vtkSmartPointer<vtkPoints> points, const Vec3f& center, const Vec3i& rotation, const Vec3f& size);
    Vec3f getVolSizeUi();
    void setVolSizeUi(const Vec3f& volSize);
    void getSettings(af::Settings& settings);
    void setSettings(const af::Settings& settings);
    void loadSettingsPresets();
    af::Settings& currentSettingsPreset();

    void setupConnections();

private:
    vtkSmartPointer<vtkEventQtSlotConnect> _connections;
    StepperWidget* _stepperWidget;
    // PointsQWrap _points;
    // vtkSmartPointer<vtkPolyData> _polydata;
    MotionGraph _motionGraph;
    std::map<std::string, af::BufferVtkPoints> _points;
    std::map<std::string, vtkSmartPointer<vtkCellArray>> _cellArrays;
    std::map<std::string, af::BufferVector<Vec3i>> _faces;
    af::BufferVector<Vecui<Constants::energyMRegKNN>> _graphKnns;
    af::BufferVector<Vec2i> _correspondenceCanonToFrame;
    af::BufferObject<Vec3f> _tsdfCenter;
    std::atomic_flag _volumeFusionRunning         = ATOMIC_FLAG_INIT;
    std::atomic_flag _directReconstructionRunning = ATOMIC_FLAG_INIT;
    std::atomic_flag _directPointCloudRunning     = ATOMIC_FLAG_INIT;
    af::Stepper _volumeFusionStepper;
    std::mutex _dataLoadMutexDirectReconstr;
    Mesh _objectMesh;
    af::VolumeFusionOutput _volumeFusionResult;
    af::MeshReconstructionBuffers _meshReconstructionBuffers;
    af::PointCloudBuffers _pointCloudBuffers;
    std::map<std::string, vtkSmartPointer<vtkSphereSource>> _spheres;
    std::map<std::string, vtkSmartPointer<vtkPolyData>> _polydatas;
    std::map<std::string, vtkSmartPointer<vtkActor>> _actors;
    std::map<std::string, QFuture<void>> _threads;
    std::map<std::string, QFutureWatcher<void>> _watchers;
    std::vector<af::Settings> _settingsPresets;
    af::Settings _settings;
};

#endif