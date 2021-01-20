

#include <QtGui/QSurfaceFormat>
#include <QtWidgets/QApplication>

#include "af/ui/MainWindow.h"
#include "af/VolumeFusion.h"

int main(int argc, char* argv[]) {
    // needed to ensure appropriate OpenGL context is created for VTK rendering.
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());

    QApplication app(argc, argv);

    MainWindow mainwindow;
    mainwindow.show();

    return app.exec();
}
