// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <typename>
void element_cast(...) noexcept = delete;
}
DPL_DEFAULT_NAMESPACE_END