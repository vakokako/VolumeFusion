#ifndef POINTSQWRAP_H
#define POINTSQWRAP_H

#include <vtkSmartPointer.h>
#include <memory>

class vtkPoints;
class ModifySignaler;

class PointsQWrap {
public:
    explicit PointsQWrap();
    PointsQWrap(const PointsQWrap& arr);

    void resize(unsigned int newSize);
    unsigned int size();
    void* data();
    vtkSmartPointer<vtkPoints> points() { return _points; }
    void modified();
    std::shared_ptr<ModifySignaler> signaler() { return _signaler; }


private:
    vtkSmartPointer<vtkPoints> _points;
    std::shared_ptr<ModifySignaler> _signaler;
};


#endif