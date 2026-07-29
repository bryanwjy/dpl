// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/unsigned_integral_type.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename>
void from_bitset(...) noexcept = delete;

template <typename T, typename U = void>
struct from_bitset_t : public basic_operation_base<from_bitset_t<T, U>> {
    using operation_base<from_bitset_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<from_bitset_t<T, U>> {
    static consteval void operator()(auto&&, auto&&...) noexcept
    requires ((same_as<void, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <simd_mask T>
struct canonical_impl<from_bitset_t<T>> :
    canonical_impl<from_bitset_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {
};

template <simd_abi A>
struct canonical_impl<from_bitset_t<A>> {
private:
    template <size_t W>
    requires fixed_width_abi<A> && (__DPL has_single_bit(W))
    using deduced_mask DPL_NODEBUG = make_canonical_mask_t<
        unsigned_integral_type_t<(A::size * char_bit_v / W)>, A>;

public:
    template <size_t W>
    requires requires { typename deduced_mask<W>; } &&
        cpo_invocable<from_bitset_t<deduced_mask<W>>, bitset<W>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_mask<W> operator()(bitset<W> data) noexcept {
        return from_bitset_t<deduced_mask<W>>::operator()(data);
    }
};

template <typename T, different_from<void> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct canonical_impl<from_bitset_t<T, U>> {
private:
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_mask_t<E, A> operator()(
        bitset<simd_abi_traits<A, E>::size> data) noexcept
    requires fixed_width_abi<A> &&
        requires { from_bitset<E>(internal::abi<A>, data); }
    {
        return from_bitset<E>(internal::abi<A>, data);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <typename T, typename U = void>
inline constexpr internal::from_bitset_t<T, U> from_bitset{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
