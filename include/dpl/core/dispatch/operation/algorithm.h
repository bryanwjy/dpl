// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/operation/base.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
class algorithm_base : public operation_base<D> {};

template <typename D>
class reduction_base : public algorithm_base<D> {};

template <typename D>
class prefix_scan_base : public algorithm_base<D> {};

template <typename D>
class exclusive_scan_base : public prefix_scan_base<D> {};

template <typename D>
class inclusive_scan_base : public prefix_scan_base<D> {};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
