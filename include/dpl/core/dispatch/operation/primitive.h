// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/operation/base.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
class primitive_operation_base : public operation_base<D> {};

template <typename D>
class arithmetic_base : public primitive_operation_base<D> {};

template <typename D>
class bit_manipulation_base : public primitive_operation_base<D> {};

template <typename D>
class bitwise_base : public primitive_operation_base<D> {};

template <typename D>
class comparison_base : public primitive_operation_base<D> {};

template <typename D>
class logical_base : public primitive_operation_base<D> {};

template <typename D>
class logical_reduction_base : public logical_base<D> {};

template <typename D>
class cast_operation_base : public primitive_operation_base<D> {};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
