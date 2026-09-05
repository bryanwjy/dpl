// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fm_broadcast.h"
#include "dpl/core/math/fma/fnmacc.h"
#if !DPL_MODULES
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/operations/arithmetic/multiply.h"
#  include "dpl/core/operations/arithmetic/subtract.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES nmulacc_t :
    public math_operation_base<nmulacc_t>,
    public maskable_accumulation_base<nmulacc_t> {
    using operation_base<nmulacc_t>::operator();
    using maskable_accumulation_base<nmulacc_t>::operator();
};

template <>
struct operation_signature<nmulacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<nmulacc_t> : fm_canonical_broadcaster<nmulacc_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(__DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subtract(dx::negate(__DPL forward<AT>(a)),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)));
        }
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<nmulacc_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<nmulacc_t, AT, BT, CT> operator()(AT a,
        simd_mask_type_t<cpo_result_t<nmulacc_t, AT, BT, CT>> mask, BT b,
        CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::negate(a), dx::multiply(b, c));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<nmulacc_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M&& mask, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc( __DPL forward<AT>(a), __DPL forward<M>(mask),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subtract(a, __DPL forward<M>(mask), dx::negate(a),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<nmulacc_t, AT, BT, CT> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M mask, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc( __DPL forward<AT>(a), mask, __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subtract(a, mask, dx::negate(a),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)));
        }
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<nmulacc_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<nmulacc_t, AT, BT, CT> operator()(
        dx::zero_t zero,
        simd_mask_type_t<cpo_result_t<nmulacc_t, AT, BT, CT>> mask, AT a, BT b,
        CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::negate(a), dx::multiply(b, c));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<nmulacc_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& a, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(zero, __DPL forward<M>(mask),
                __DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subtract(zero, __DPL forward<M>(mask),
                dx::negate(__DPL forward<AT>(a)),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<nmulacc_t, AT, BT, CT> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, AT&& a, BT&& b,
        CT&& c) noexcept(canonical_vector<AT> && canonical_vector<BT> &&
        canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(zero, mask, __DPL forward<AT>(a),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subtract(zero, mask, dx::negate(__DPL forward<AT>(a)),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)));
        }
    }

    using fm_canonical_broadcaster<nmulacc_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::nmulacc_t nmulacc{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
