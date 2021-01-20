#include <af/ui/ModifySignaler.h>
#include <af/ui/ModifySignalerQt.h>

ModifySignaler::ModifySignaler() : _signalerQt(std::make_unique<ModifySignalerQt>()) {}
ModifySignaler::ModifySignaler(const ModifySignaler& other) : _signalerQt(std::make_unique<ModifySignalerQt>(*other._signalerQt)) {}
ModifySignaler::~ModifySignaler() {}

void ModifySignaler::modified() {
    _signalerQt->modified();
}