#ifndef MAINWINDOWUI_H
#define MAINWINDOWUI_H

#include <vtkActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkPointPicker.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkVertexGlyphFilter.h>

#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <type_traits>

#include "QVTKOpenGLNativeWidget.h"
#include "af/eigen_extension.h"
#include "af/ui/QSpinBoxTuple.h"
#include "af/ui/StepperWidget.h"

class MainWindowUI {
public:
    QWidget* widgetCentral;
    QHBoxLayout* layoutHMain;
    QVBoxLayout* layoutVMenu;
    QVTKOpenGLNativeWidget* qvtkWidget;

    QWidget* mainTabWidget     = new QWidget;
    QVBoxLayout* mainTabLayout = createVLayout(mainTabWidget);
    QComboBox* comboboxPresets = [&] {
        auto combobox = new QComboBox();
        combobox->addItem("upperbody");
        combobox->addItem("multi");
        mainTabLayout->addWidget(combobox);
        return combobox;
    }();
    QPushButton* pushSaveSettings              = createPushButton("save settings", mainTabLayout);
    QLineEdit* lineEditData                    = createLineEdit("Data folder:", mainTabLayout);
    QSpinBox* spinboxCamerasCount              = createSpinBox("cameras count", mainTabLayout);
    QLineEdit* lineEditDepthFilesPattern       = createLineEdit("depth files pattern:", mainTabLayout);
    QDoubleSpinBox* spinboxDepthFilter         = createDoubleSpinBox("depth filter", mainTabLayout, 3);
    QSpinboxTuple<int, 2>* spinboxFramesRange  = createSpinBoxTuple<int, 2>("Frames:", mainTabLayout, 0, 100000);
    QDoubleSpinBox* spinboxDoubleWeightDepth   = createDoubleSpinBox("energy depth weight:", mainTabLayout);
    QDoubleSpinBox* spinboxDoubleWeightMReg    = createDoubleSpinBox("energy mreg weight:", mainTabLayout);
    QDoubleSpinBox* spinboxDoubleEnergyMinStep = createDoubleSpinBox("energy error max:", mainTabLayout, 5, 0.0001);
    QSpinBox* spinboxIcpIterations             = createSpinBox("icp iterations", mainTabLayout);
    QSpinboxTuple<float, 3>* spinboxDoubleVolSize =
        createPreciseSpinBoxTuple<float, 3>("volume size:", mainTabLayout, 2, 0.1, 20., 0.1);
    QSpinboxTuple<int, 3>* spinboxVolDim = createSpinBoxTuple<int, 3>("volume dimensions:", mainTabLayout, 1, 100000);
    QSpinboxTuple<float, 3>* spinboxDoubleVolCenter =
        createPreciseSpinBoxTuple<float, 3>("volume center:", mainTabLayout, 2, -20., 20., 0.1);
    QSpinboxTuple<int, 3>* spinboxVolRotation    = createSpinBoxTuple<int, 3>("volume rotation:", mainTabLayout, -1000, 1000);
    QDoubleSpinBox* spinboxDoubleTsdfDelta       = createDoubleSpinBox("tsdf delta:", mainTabLayout);
    QDoubleSpinBox* spinboxDoubleGraphRadius     = createDoubleSpinBox("graph radius:", mainTabLayout);
    QDoubleSpinBox* spinboxDoubleGraphSampleDist = createDoubleSpinBox("graph sample distance:", mainTabLayout);
    QDoubleSpinBox* spinboxDoubleCorrespThreshDist =
        createDoubleSpinBox("correspondence distance threshold:", mainTabLayout, 4, 0.001);
    QCheckBox* checkIsMeshWarped                  = createCheckBox("is mesh warped", mainTabLayout);
    QPushButton* pushStartVolume             = createPushButton("Start VolumeFusion", mainTabLayout);
    QPushButton* pushStartMeshReconstruction = createPushButton("Start Direct Mesh Reconstruction", mainTabLayout);
    QPushButton* pushStartPointCloudStream   = createPushButton("Start Direct Point cloud stream", mainTabLayout);
    QCheckBox* checkDisplayMeshCanon         = createCheckBox("display canon mesh", mainTabLayout);
    QCheckBox* checkDisplayPointsCanon       = createCheckBox("display canon points", mainTabLayout);
    QCheckBox* checkDisplayPointsWarped      = createCheckBox("display canon points warped", mainTabLayout);
    QCheckBox* checkDisplayGraph             = createCheckBox("display graph", mainTabLayout);
    QCheckBox* checkDisplayGraphRadiuses     = createCheckBox("display graph radiuses", mainTabLayout);
    QCheckBox* checkDisplayCorrespondences   = createCheckBox("display correspondences", mainTabLayout);
    QCheckBox* checkDisplayPointsFrame       = createCheckBox("display frame points", mainTabLayout);
    QCheckBox* checkDisplayTsdfBox           = createCheckBox("display tsdf box", mainTabLayout);
    QSpinBox* spinboxFrameBreak              = createSpinBox("frame break", mainTabLayout, -1, 10000);
    QPushButton* pushWarpMesh                = createPushButton("warp mesh", mainTabLayout);
    QComboBox* comboboxMeshType = [&] {
        auto combobox = new QComboBox();
        combobox->addItem("canon");
        combobox->addItem("warped");
        combobox->setCurrentIndex(0);
        mainTabLayout->addWidget(combobox);
        return combobox;
    }();
    QLineEdit* lineEditMeshName              = createLineEdit("mesh file name:", mainTabLayout, "/canonical/frame-000000.canonical.vako");
    QPushButton* pushSaveMesh                = createPushButton("save mesh", mainTabLayout);

    QWidget* debugTabWidget                         = new QWidget;
    QVBoxLayout* debugTabLayout                     = createVLayout(debugTabWidget);
    QCheckBox* checkBilateralFiltr                  = createCheckBox("bilateral filtering", debugTabLayout);
    QSpinboxTuple<float, 4>* spinboxBilateralParams = createPreciseSpinBoxTuple<float, 4>("", debugTabLayout, 6);
    QCheckBox* checkIntegrateTsdf                   = createCheckBox("integrate tsdf", debugTabLayout);
    QCheckBox* checkUpdateGraph                     = createCheckBox("update graph", debugTabLayout);
    QCheckBox* checkUseCholesky                     = createCheckBox("use cholesky", debugTabLayout);
    QCheckBox* checkUseCopyHack                     = createCheckBox("use copy hack", debugTabLayout);
    QSlider* sliderPointSize                        = createSlider("point size:", debugTabLayout, 1, 12);
    QSpinboxTuple<float, 2>* spinboxNormsExcl       = createPreciseSpinBoxTuple<float, 2>("exclude norms:", debugTabLayout, 2);
    QCheckBox* checkDisplayVisualization            = createCheckBox("Display visualization", debugTabLayout);
    QPushButton* pushTestVolume                     = createPushButton("Test volume fusion", debugTabLayout);
    QPushButton* pushTestDirect                     = createPushButton("Test direct", debugTabLayout);

    void setupUi(QMainWindow* MainWindow) {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(3000, 2000);

        widgetCentral = new QWidget(MainWindow);
        widgetCentral->setObjectName(QString::fromUtf8("widgetCentral"));

        layoutHMain = new QHBoxLayout(widgetCentral);
        layoutHMain->setObjectName(QString::fromUtf8("layoutHMain"));

        qvtkWidget = new QVTKOpenGLNativeWidget(widgetCentral);
        qvtkWidget->setObjectName(QString::fromUtf8("qvtkWidget"));

        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(qvtkWidget->sizePolicy().hasHeightForWidth());
        qvtkWidget->setSizePolicy(sizePolicy);
        layoutHMain->addWidget(qvtkWidget);

        MainWindow->setCentralWidget(widgetCentral);

        layoutVMenu = new QVBoxLayout();
        layoutVMenu->setAlignment(Qt::AlignTop);
        layoutHMain->addWidget(qvtkWidget, 1);
        layoutHMain->addLayout(layoutVMenu);

        std::vector<QScrollArea*> scrollAreas(2);
        for (auto& area : scrollAreas) {
            area = new QScrollArea;
        }
        scrollAreas[0]->setWidget(mainTabWidget);
        scrollAreas[1]->setWidget(debugTabWidget);

        QTabWidget* tabWidget = new QTabWidget;
        tabWidget->addTab(scrollAreas[0], "Main");
        tabWidget->addTab(scrollAreas[1], "Debug");

        layoutVMenu->addWidget(tabWidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    }  // setupUi

    void retranslateUi(QMainWindow* MainWindow) { MainWindow->setWindowTitle("VolumeFusion"); }  // retranslateUi

protected:
    template<typename Layout>
    QCheckBox* createCheckBox(const std::string& name, Layout* layout, bool checked = true) {
        QCheckBox* checkBox = new QCheckBox(name.c_str());
        checkBox->setChecked(checked);
        layout->addWidget(checkBox);
        return checkBox;
    }

    template<typename Layout>
    QLineEdit* createLineEdit(const std::string& name, Layout* layout, const std::string& value = "") {
        auto hLayout  = new QHBoxLayout();
        auto label    = new QLabel(name.c_str());
        auto lineEdit = new QLineEdit(value.c_str());
        hLayout->addWidget(label);
        hLayout->addWidget(lineEdit);
        layout->addLayout(hLayout);
        return lineEdit;
    }

    QVBoxLayout* createVLayout(QWidget* widget = nullptr, Qt::Alignment alignment = Qt::AlignTop) {
        QVBoxLayout* vLayout = new QVBoxLayout();
        vLayout->setAlignment(alignment);
        vLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
        if (widget)
            widget->setLayout(vLayout);
        return vLayout;
    }

    template<typename Widget, typename Layout>
    Widget* createWidget(Layout* layout) {
        Widget* widget = new Widget();
        layout->addWidget(widget);
        return widget;
    }

    template<typename Layout>
    QSlider* createSlider(const std::string& name,
                          Layout* layout,
                          int min                     = 0,
                          int max                     = 99,
                          Qt::Orientation orientation = Qt::Horizontal) {
        auto hLayout = new QHBoxLayout();
        auto label   = new QLabel(name.c_str());
        auto slider  = new QSlider(orientation);
        slider->setRange(min, max);
        hLayout->addWidget(label);
        hLayout->addWidget(slider);
        layout->addLayout(hLayout);
        return slider;
    }

    template<typename Layout>
    QSpinBox* createSpinBox(const std::string& name, Layout* layout, int min = 0, int max = 99) {
        auto hLayout = new QHBoxLayout();
        auto label   = new QLabel(name.c_str());
        auto spinbox = new QSpinBox();
        spinbox->setRange(min, max);
        hLayout->addWidget(label);
        hLayout->addWidget(spinbox);
        layout->addLayout(hLayout);

        return spinbox;
    }

    template<typename Layout>
    QDoubleSpinBox* createDoubleSpinBox(const std::string& name,
                                        Layout* layout,
                                        int decimals = 5,
                                        double step  = 0.1,
                                        double min   = 0.,
                                        double max   = 100.) {
        auto hLayout = new QHBoxLayout();
        auto label   = new QLabel(name.c_str());
        auto spinbox = new QDoubleSpinBox();
        spinbox->setRange(min, max);
        spinbox->setDecimals(decimals);
        spinbox->setSingleStep(step);
        hLayout->addWidget(label);
        hLayout->addWidget(spinbox);
        layout->addLayout(hLayout);

        return spinbox;
    }

    template<typename Layout>
    QPushButton* createPushButton(const std::string& name, Layout* layout) {
        auto button = new QPushButton(name.c_str());
        layout->addWidget(button);
        return button;
    }

    template<typename T, std::size_t N, typename Layout>
    QSpinboxTuple<T, N>* createSpinBoxTuple(const std::string& name, Layout* layout, T min = 0, T max = 100, T singleStep = 1) {
        auto hLayout = new QHBoxLayout();
        auto label   = new QLabel(name.c_str());
        auto tuple   = new QSpinboxTuple<T, N>();
        tuple->setRange(min, max);
        tuple->setSingleStep(singleStep);

        hLayout->addWidget(label);
        for (std::size_t i = 0; i < N; ++i) {
            hLayout->addWidget(tuple->at(i));
        }
        layout->addLayout(hLayout);
        return tuple;
    }

    template<typename T, std::size_t N, typename Layout>
    QSpinboxTuple<T, N>* createPreciseSpinBoxTuple(const std::string& name,
                                                   Layout* layout,
                                                   int decimals = 5,
                                                   T min        = 0,
                                                   T max        = 100,
                                                   T singleStep = 1) {
        auto tuple = createSpinBoxTuple<T, N>(name, layout, min, max, singleStep);
        tuple->setDecimals(decimals);
        return tuple;
    }

    template<typename Layout>
    QWidget* createTabWidget(std::initializer_list<std::pair<QWidget*, std::string>> tabs, Layout* layout = nullptr) {
        QTabWidget* tabWidget = new QTabWidget;
        for (auto&& tab : tabs) {
            tabWidget->addTab(tab.first, tab.second.c_str());
        }
        if (layout) {
            layout->addWidget(tabWidget);
        }
        return tabWidget;
    }
};

class InteractorStyleSelect : public vtkInteractorStyleTrackballCamera {
public:
    static InteractorStyleSelect* New() { VTK_STANDARD_NEW_BODY(InteractorStyleSelect); }
    vtkTypeMacro(InteractorStyleSelect, vtkInteractorStyleTrackballCamera)

        InteractorStyleSelect() {
        this->Move        = false;
        this->PointPicker = vtkSmartPointer<vtkPointPicker>::New();

        // Setup ghost glyph
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        points->InsertNextPoint(0, 0, 0);
        this->MovePolyData = vtkSmartPointer<vtkPolyData>::New();
        this->MovePolyData->SetPoints(points);
        this->MoveGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        this->MoveGlyphFilter->SetInputData(this->MovePolyData);
        this->MoveGlyphFilter->Update();

        this->MoveMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        this->MoveMapper->SetInputConnection(this->MoveGlyphFilter->GetOutputPort());

        this->MoveActor = vtkSmartPointer<vtkActor>::New();
        this->MoveActor->SetMapper(this->MoveMapper);
        this->MoveActor->VisibilityOff();
        this->MoveActor->GetProperty()->SetPointSize(30);
        this->MoveActor->GetProperty()->SetColor(1, 0, 0);
        this->MoveActor->GetProperty()->SetOpacity(0.4);
    }

    void OnMiddleButtonDown() {
        // Get the selected point
        int x = this->Interactor->GetEventPosition()[0];
        int y = this->Interactor->GetEventPosition()[1];
        this->FindPokedRenderer(x, y);

        int picked = this->PointPicker->Pick(this->Interactor->GetEventPosition()[0], this->Interactor->GetEventPosition()[1],
                                             0,  // always zero.
                                             this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());

        if (picked && this->PointPicker->GetActor() == DataActor) {
            // this->StartPan();
            this->MoveActor->VisibilityOn();
            this->Move          = true;
            this->SelectedPoint = this->PointPicker->GetPointId();

            std::cout << "selected point " << this->SelectedPoint << std::endl;

            double p[3];
            // this->Data->GetProp
            this->Data->GetPoint(this->SelectedPoint, p);
            std::cout << "p: " << p[0] << " " << p[1] << " " << p[2] << std::endl;
            this->MoveActor->SetPosition(p);

            this->GetCurrentRenderer()->AddActor(this->MoveActor);
            // this->InteractionProp = this->MoveActor;
        }
        vtkInteractorStyleTrackballCamera::OnMiddleButtonDown();
    }
    void OnMouseMove() { vtkInteractorStyleTrackballCamera::OnMouseMove(); }
    void OnLeftButtonDown() { vtkInteractorStyleTrackballCamera::OnLeftButtonDown(); }
    void OnLeftButtonUp() { vtkInteractorStyleTrackballCamera::OnLeftButtonUp(); }
    //   void OnMiddleButtonDown() { vtkInteractorStyleTrackballCamera::OnMiddleButtonDown(); }
    void OnMiddleButtonUp() { vtkInteractorStyleTrackballCamera::OnMiddleButtonUp(); }
    void OnRightButtonDown() { vtkInteractorStyleTrackballCamera::OnRightButtonDown(); }
    void OnRightButtonUp() { vtkInteractorStyleTrackballCamera::OnRightButtonUp(); }
    void OnMouseWheelForward() { vtkInteractorStyleTrackballCamera::OnMouseWheelForward(); }
    void OnMouseWheelBackward() { vtkInteractorStyleTrackballCamera::OnMouseWheelBackward(); }
    //@}

    // These methods for the different interactions in different modes
    // are overridden in subclasses to perform the correct motion. Since
    // they might be called from OnTimer, they do not have mouse coord parameters
    // (use interactor's GetEventPosition and GetLastEventPosition)
    void Rotate() { vtkInteractorStyleTrackballCamera::Rotate(); }
    void Spin() { vtkInteractorStyleTrackballCamera::Spin(); }
    void Pan() { vtkInteractorStyleTrackballCamera::Pan(); }
    void Dolly() { vtkInteractorStyleTrackballCamera::Dolly(); }

    vtkPolyData* Data;
    vtkActor* DataActor;
    vtkPolyData* GlyphData;

    vtkSmartPointer<vtkPolyDataMapper> MoveMapper;
    vtkSmartPointer<vtkActor> MoveActor;
    vtkSmartPointer<vtkPolyData> MovePolyData;
    vtkSmartPointer<vtkVertexGlyphFilter> MoveGlyphFilter;

    vtkSmartPointer<vtkPointPicker> PointPicker;

    bool Move;
    vtkIdType SelectedPoint;
};

#endif