// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/details/type_traits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

// Cheaper than invocable, avoids instantiating __DPL invoke & friends
template <typename Cpo, typename... Ts>
concept cpo_invocable =
    requires(Cpo const op) { op(internal::declarg<Ts>()...); };

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
