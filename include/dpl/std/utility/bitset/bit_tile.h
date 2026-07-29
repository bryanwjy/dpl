// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/ignore.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/has_unique_object_representations.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t N, size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset<N * W> bit_tile(
    bitset<W> const& set) noexcept {
    return [&]<size_t... Is>(index_sequence<Is...>) {
        return bitset<N * W>((__DPL ignore = Is, set)...);
    }(make_index_sequence<N>{});
}

__DPL_DEFAULT_NAMESPACE_END
