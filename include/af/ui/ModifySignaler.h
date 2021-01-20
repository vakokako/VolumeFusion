#ifndef MODIFYSIGNALER_H
#define MODIFYSIGNALER_H

#include <memory>

class ModifySignalerQt;

class ModifySignaler {
public:
    explicit ModifySignaler();
    ModifySignaler(const ModifySignaler& other);
    ~ModifySignaler();

    void modified();

    ModifySignalerQt* qt() { return _signalerQt.get(); }
    const ModifySignalerQt* qt() const { return _signalerQt.get(); }

private:
    std::unique_ptr<ModifySignalerQt> _signalerQt;
};


#endif