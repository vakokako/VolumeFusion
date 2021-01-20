#include <af/ui/PointsQWrap.h>
#include <af/ui/ModifySignaler.h>
#include <vtkPoints.h>

PointsQWrap::PointsQWrap() : _points(vtkSmartPointer<vtkPoints>::New()), _signaler(std::make_shared<ModifySignaler>()) {}

PointsQWrap::PointsQWrap(const PointsQWrap &arr) : _points(arr._points), _signaler(std::make_shared<ModifySignaler>()) {}

void PointsQWrap::resize(unsigned int newSize) {
    unsigned int pointsCount = newSize;
    _points->Resize(pointsCount);
    _points->SetNumberOfPoints(pointsCount);
}

unsigned int PointsQWrap::size() { return _points->GetNumberOfPoints(); }

void *PointsQWrap::data() { return (_points->GetVoidPointer(0)); }

void PointsQWrap::modified() {
    _points->Modified();
    _signaler->modified();
}