// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void to_bitset(...) noexcept = delete;

struct to_bitset_t : private basic_operation_base<to_bitset_t> {
    using operation_base<to_bitset_t>::operator();
};

template <>
struct operation_signature<to_bitset_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_to_bitset = requires {
    { to_bitset(internal::abi<T>, internal::declarg<T>()) } -> bitset_type;
};

template <>
struct canonical_impl<to_bitset_t> {

    template <fixed_width_abi A, simd_element_for<A> E>
    requires unqualified_canonical_to_bitset<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bitset<simd_abi_traits<A, E>::size> operator()(
        basic_mask<E, A> val) noexcept {
        return to_bitset(internal::abi<A>, val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::to_bitset_t to_bitset{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
