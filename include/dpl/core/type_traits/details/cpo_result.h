// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/declarg.h"

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace datapar::internal {

// Cheaper than invoke_result_t, avoids instantiating __DPL invoke & friends
template <typename D, typename... Ts>
using cpo_result_t DPL_NODEBUG =
    decltype(internal::declarg<D const&>()(internal::declarg<Ts>()...));

template <typename D, typename... Ts>
struct cpo_result {
    using type DPL_NODEBUG = cpo_result_t<D, Ts...>;
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
