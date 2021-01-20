#ifndef STREAMBUFFERQTPOINTS_H
#define STREAMBUFFERQTPOINTS_H

#include <vtkPoints.h>
#include <vtkSmartPointer.h>

#include "af/eigen_extension.h"
#include "af/ui/BufferSignaling.h"

namespace af {

class BufferVtkPoints : public SignalingStreamBuffer<Vec3f> {
public:
    BufferVtkPoints(bool silent = false) : _points(vtkSmartPointer<vtkPoints>::New()), _silent(silent) {}

    void load(const Vec3f* data, std::size_t size) final { load(true, data, size); }
    void loadFromDevice(const Vec3f* data, std::size_t size) final { load(false, data, size); }
    void modified() final {
        _points->Modified();
        SignalingStreamBuffer<Vec3f>::modified();
    }

    bool isSilent() const { return _silent; }
    void setSilent(bool silent) { _silent = silent; }

    vtkSmartPointer<vtkPoints> points() { return _points; }

private:
    void load(bool isHost, const Vec3f* data, std::size_t size);

    vtkSmartPointer<vtkPoints> _points;
    bool _silent;
};

}  // namespace af

#endif