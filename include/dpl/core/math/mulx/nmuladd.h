// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fm_broadcast.h"
#include "dpl/core/math/fma/fnmadd.h"
#if !DPL_MODULES
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/operations/arithmetic/multiply.h"
#  include "dpl/core/operations/arithmetic/subtract.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES nmuladd_t :
    public math_operation_base<nmuladd_t>,
    public maskable_accumulation_base<nmuladd_t> {
    using operation_base<nmuladd_t>::operator();
    using maskable_accumulation_base<nmuladd_t>::operator();
};

template <>
struct operation_signature<nmuladd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<nmuladd_t> : fm_canonical_broadcaster<nmuladd_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(__DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subtract(dx::negate(__DPL forward<CT>(c)),
                dx::multiply(__DPL forward<AT>(a), __DPL forward<BT>(b)));
        }
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<nmuladd_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<nmuladd_t, AT, BT, CT> operator()(AT a,
        simd_mask_type_t<cpo_result_t<nmuladd_t, AT, BT, CT>> mask, BT b,
        CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::negate(c), dx::multiply(a, b));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<nmuladd_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M&& mask, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd( __DPL forward<AT>(a), __DPL forward<M>(mask),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subtract(a, __DPL forward<M>(mask),
                dx::negate(__DPL forward<CT>(c)),
                dx::multiply(a, __DPL forward<BT>(b)));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<nmuladd_t, AT, BT, CT> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M mask, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd( __DPL forward<AT>(a), mask, __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subtract(a, mask, dx::negate(__DPL forward<CT>(c)),
                dx::multiply(a, __DPL forward<BT>(b)));
        }
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<nmuladd_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<nmuladd_t, AT, BT, CT> operator()(
        dx::zero_t zero,
        simd_mask_type_t<cpo_result_t<nmuladd_t, AT, BT, CT>> mask, AT a, BT b,
        CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::negate(c), dx::multiply(a, b));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<nmuladd_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& a, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(zero, __DPL forward<M>(mask),
                __DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subtract(zero, __DPL forward<M>(mask),
                dx::negate(__DPL forward<CT>(c)),
                dx::multiply(__DPL forward<AT>(a), __DPL forward<BT>(b)));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<nmuladd_t, AT, BT, CT> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, AT&& a, BT&& b,
        CT&& c) noexcept(canonical_vector<AT> && canonical_vector<BT> &&
        canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(zero, mask, __DPL forward<AT>(a),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subtract(zero, mask, dx::negate(__DPL forward<CT>(c)),
                dx::multiply(__DPL forward<AT>(a), __DPL forward<BT>(b)));
        }
    }

    using fm_canonical_broadcaster<nmuladd_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::nmuladd_t nmuladd{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
