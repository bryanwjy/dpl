// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/operation/base.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/broadcastable_to.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
struct bwand_t;
struct select_t;

namespace fwd {

struct fbwand_t : private operation_base<bwand_t> {
    using operation_base<bwand_t>::operator();
};

struct fselect_t : private operation_base<select_t> {
    using operation_base<select_t>::operator();
};

template <typename T, typename U = ignore_t>
struct fbroadcast_t : private operation_base<broadcast_t<T, U>> {
    using operation_base<broadcast_t<T, U>>::operator();
};

inline constexpr fbwand_t bwand;
inline constexpr fselect_t select;
template <typename T, typename U = ignore_t>
inline constexpr fbroadcast_t<T, U> broadcast;
} // namespace fwd
} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
