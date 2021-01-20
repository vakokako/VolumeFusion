#ifndef MODIFYSIGNALERQT_H
#define MODIFYSIGNALERQT_H


#include <QtCore/QObject>

class ModifySignalerQt : public QObject {
    Q_OBJECT
public:
    explicit ModifySignalerQt(QObject* parent = 0) : QObject(parent) {}
    ModifySignalerQt(const ModifySignalerQt& other) : QObject(other.parent()) {}

public slots:
    void modified() {
        emit modifiedSignal();
    }

signals:
    void modifiedSignal();

};

Q_DECLARE_METATYPE(ModifySignalerQt)


#endif