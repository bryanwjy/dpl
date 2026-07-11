// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/details/declarg.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

// Cheaper than invocable, avoids instantiating __DPL invoke & friends
template <typename Cpo, typename... Ts>
concept cpo_invocable =
    requires(Cpo const op) { op(internal::declarg<Ts>()...); };

} // namespace datapar::internal
__DPL_DEFAULT_NAMESPACE_END
