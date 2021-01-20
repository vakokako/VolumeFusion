#ifndef STEPPERWIDGET_H
#define STEPPERWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>

#include <map>
#include <string>

#include "af/Stepper.h"

class StepperWidget : public QWidget {
    Q_OBJECT
public:
    explicit StepperWidget(af::Stepper& stepper, QWidget* parent = nullptr) : QWidget(parent), _stepper(stepper) {
        auto mainLayout = new QVBoxLayout;
        mainLayout->setAlignment(Qt::AlignTop);

        _checkBoxEnableStepper = new QCheckBox("enable stepping");
        _checkBoxEnableStepper->setChecked(_stepper.isEnabled());
        connect(_checkBoxEnableStepper, &QCheckBox::stateChanged, [this](int state) {
            if (state == Qt::PartiallyChecked)
                return;

            bool enable = state == Qt::Checked ? true : false;
            _stepper.enable(enable);
        });

        auto pushButtonStep = new QPushButton("step");
        connect(pushButtonStep, &QPushButton::pressed, [this]() {
            updateStepSwitchesUi();
            _stepper.step();
        });

        auto hLayout = new QHBoxLayout;
        hLayout->addWidget(_checkBoxEnableStepper);
        hLayout->addWidget(pushButtonStep);

        mainLayout->addLayout(hLayout);

        setLayout(mainLayout);

        updateStepSwitchesUi();
    }

    ~StepperWidget() {}

    void updateStepSwitchesUi() {
        _checkBoxEnableStepper->setChecked(_stepper.isEnabled());
        for (auto&& stepSwitch : _stepper.stepsSwitches()) {
            const std::string& stepName = stepSwitch.first;

            if (_checkBoxes.count(stepName)) {
                _checkBoxes[stepName]->setChecked(stepSwitch.second);
                continue;
            }

            auto checkBox = new QCheckBox(stepName.c_str());
            connect(checkBox, &QCheckBox::stateChanged, [stepName, this](int state) {
                if (state == Qt::PartiallyChecked)
                    return;
                bool enabled = state == Qt::Checked ? true : false;
                _stepper.enableStep(stepName, enabled);
            });
            checkBox->setObjectName(stepName.c_str());
            checkBox->setChecked(stepSwitch.second);
            layout()->addWidget(checkBox);
            _checkBoxes[stepName] = checkBox;
        }
    }

    void setEnabled(bool enabled) {
        _checkBoxEnableStepper->setChecked(enabled ? Qt::Checked : Qt::Unchecked);
    }

public slots:
    void stepSwitchChanged() {

    }

private:
    QCheckBox* _checkBoxEnableStepper;
    std::map<std::string, QCheckBox*> _checkBoxes;
    af::Stepper& _stepper;
};

#endif