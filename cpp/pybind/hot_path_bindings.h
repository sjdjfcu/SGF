#pragma once

#include <pybind11/pybind11.h>

namespace sgf_pybind {

void RegisterHotPathBindings(pybind11::module_ &module);

}  // namespace sgf_pybind
