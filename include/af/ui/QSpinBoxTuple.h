#ifndef QSPINBOXTUPLE_H
#define QSPINBOXTUPLE_H

#include <QtWidgets/QDoubleSpinBox>
#include <array>
#include <type_traits>

#include "af/eigen_extension.h"

template<typename T, std::size_t N>
class QSpinboxTuple {
public:
    using Widget = std::conditional_t<std::is_floating_point_v<T>, QDoubleSpinBox, QSpinBox>;
    std::array<Widget*, N> _tuple;

    explicit QSpinboxTuple(QWidget* parent = nullptr) {
        for (auto&& spinbox : _tuple) {
            spinbox = new Widget(parent);
        }
    }

    template<typename T_ = T, typename = std::enable_if_t<std::is_floating_point_v<T_>>>
    void setDecimals(int decimals) {
        for (auto&& spinbox : _tuple) {
            spinbox->setDecimals(decimals);
        }
    }

    void setSingleStep(T step) {
        for (auto&& spinbox : _tuple) {
            spinbox->setSingleStep(step);
        }
    }

    void setRange(T min, T max) {
        for (auto&& spinbox : _tuple) {
            spinbox->setRange(min, max);
        }
    }

    Vec<T, N> getValues() const {
        Vec<T, N> values;
        for (std::size_t i = 0; i < N; ++i) {
            values[i] = _tuple[i]->value();
        }
        return values;
    }

    void setValues(const Vec<T, N>& values) {
        for (std::size_t i = 0; i < N; ++i) {
            _tuple[i]->setValue(values[i]);
        }
    }

    Widget* at(std::size_t index) { return _tuple[index]; }
};

#endif