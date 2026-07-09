// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace ranges {
inline namespace cpo {
DPL_EXPORT template <size_t I>
inline constexpr details::concepts::get_element_t<I> get_element{};
}
} // namespace ranges

DPL_EXPORT template <typename T>
concept tuple_like = details::concepts::tuple_like<T>;

DPL_DEFAULT_NAMESPACE_END
