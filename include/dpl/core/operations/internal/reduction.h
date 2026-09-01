// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/bitwise/bwxor.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/type_traits/decay.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/type_traits/sequence.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename F, typename T>
concept reduction_operator_for =
    simd_vector<T> && regular_invocable<F, T const&, T const&> &&
    simd_vector<invoke_result_t<F, T const&, T const&>> &&
    same_as<invoke_result_t<F, T const&, T const&>, decay_t<T>>;

template <integral_constant_like N, fixed_width_vector T,
    reduction_operator_for<T> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T DPL_VECTORCALL reduction(N size, T val, Op op) noexcept(
    is_nothrow_invocable_v<Op, T, T> && canonical_vector<T>) {
    constexpr auto simd_size = simd_abi_traits<T>::size();
    static_assert(simd_size >= N::value);
    return [&op]<size_t I>(this auto self, auto val, immediate<I> offset) {
        if constexpr (I >= N::value) {
            return val;
        } else {
            auto const idx = []<size_t... Is>(index_sequence<Is...>) {
                return index_sequence<(Is ^ I)...>{};
            }(iota_sequence<T>);

            return self(
                __DPL invoke(op, val, dx::permute(val, idx)), imm<I * 2>);
        }
    }(val, imm<1zu>);
}

template <simd_vector T, reduction_operator_for<T> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T DPL_VECTORCALL reduction(size_t size, T val, Op op) noexcept(
    is_nothrow_invocable_v<Op, T, T> && canonical_vector<T>) {
    constexpr auto simd_size = simd_abi_traits<T>::size();
    auto const idx = dx::lane_index<signed_canonical_vector_t<T>>();
    using idx_t = remove_const_t<decltype(idx)>;
    using sint = typename idx_t::value_type;
    for (auto offset = 1zu; offset < size; offset <<= 1) {
        auto const perm = dx::bwxor(idx, static_cast<sint>(offset));
        val = __DPL invoke(op, val, dx::permute(val, perm));
    }

    return val;
}

template <simd_vector T, reduction_operator_for<T> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T DPL_VECTORCALL reduction(T val, Op&& op) noexcept(
    is_nothrow_invocable_v<Op, T, T> && canonical_vector<T>) {
    if constexpr (fixed_width_vector<T>) {
        return internal::reduction(
            simd_abi_traits<T>::size, val, __DPL forward<Op>(op));
    } else {
        return internal::reduction(
            simd_abi_traits<T>::size(), val, __DPL forward<Op>(op));
    }
}
} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
