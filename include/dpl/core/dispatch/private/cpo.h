// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/interface.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

namespace impl {
inline namespace cpo {
template <typename D>
inline constexpr extended_impl<D> extended;
template <typename D>
inline constexpr canonical_impl<D> canonical;
template <typename D>
inline constexpr fallback_impl<D> fallback;
} // namespace cpo
} // namespace impl

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
