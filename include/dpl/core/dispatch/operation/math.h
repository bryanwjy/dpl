// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/operation/base.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
class math_operation_base : public operation_base<D> {};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
