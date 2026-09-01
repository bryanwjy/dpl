// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/dispatch/operation/base.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/broadcast_lane.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/type_traits/decay.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
struct exscan_sum_t;
namespace fwd {
struct fexscan_sum : operation_base<exscan_sum_t> {};
inline constexpr fexscan_sum exscan_sum{};
} // namespace fwd

template <typename F, typename T>
concept scan_operator_for =
    simd_vector<T> && regular_invocable<F, T const&, T const&> &&
    simd_vector<invoke_result_t<F, T const&, T const&>> &&
    same_as<invoke_result_t<F, T const&, T const&>, decay_t<T>>;

template <fixed_width_vector T, integral_constant_like Imm,
    scan_operator_for<T> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto DPL_VECTORCALL inclusive_scan(Imm, T val, Op&& op) {

    [&]<size_t I>(this auto self, immediate<I> offset) {
        using bitset_t = bitset<simd_abi_traits<T>::size>;
        constexpr auto lane_mask = __DPL apply(
            [offset]<typename... Is>(Is... idx) {
                constexpr auto S = simd_abi_traits<T>::size();
                constexpr auto set =
                    bitset_t((Is::value >= I && Is::value < Imm::value)...);
                using underlying = typename bitset_t::underlying_type;
                if constexpr (integral<underlying>) {
                    return const_mask<S, static_cast<underlying>(set)>{};
                } else {
                    return const_mask<S, set>{};
                }
            },
            iota_sequence<T>);

        auto const shifted = dx::shift_right(val, offset);
        val = dx::select(lane_mask, __DPL invoke(op, shifted, val), val);
        if constexpr (I < Imm::value) {
            self(imm<I * 2>);
        }
    }(imm<1zu>);

    return val;
}

template <simd_vector T, scan_operator_for<T> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T DPL_VECTORCALL inclusive_scan(size_t size, T val, Op op) {
    auto const iota = dx::lane_index<signed_canonical_vector_t<T>>();
    for (auto i = 0zu; i < size; i <<= 1) {
        auto const shifted = dx::shift_right(val, i);
        auto const offset = dx::broadcast_lane(iota, i);
        val = dx::select(iota >= offset, op(shifted, val), val);
    }

    return val;
}

template <simd_vector T, scan_operator_for<T> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decay_t<T>
    DPL_VECTORCALL inclusive_scan(T&& val, Op&& op) {
    using type = remove_cvref_t<T>;
    if constexpr (fixed_width_vector<type>) {
        return internal::inclusive_scan(simd_abi_traits<type>::size,
            __DPL forward<T>(val), __DPL forward<Op>(op));
    } else {
        return internal::inclusive_scan(simd_abi_traits<type>::size(),
            __DPL forward<T>(val), __DPL forward<Op>(op));
    }
}

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
