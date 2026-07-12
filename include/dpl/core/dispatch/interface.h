// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
struct extended_impl {};

template <typename D>
struct canonical_impl {};

template <typename D>
struct fallback_impl {};

template <typename D>
struct operation_signature {};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
