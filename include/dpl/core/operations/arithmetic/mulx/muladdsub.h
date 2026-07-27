// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic/addsub.h"
#include "dpl/core/operations/arithmetic/fma/fmaddsub.h"
#include "dpl/core/operations/arithmetic/multiply.h"
#if !DPL_MODULES
#  include "dpl/core/dispatch/concepts/operation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES muladdsub_t :
    public arithmetic_base<muladdsub_t>,
    public maskable_accumulation_base<muladdsub_t>,
    public ternary_broadcastable_operation<muladdsub_t> {
    using operation_base<muladdsub_t>::operator();
    using maskable_accumulation_base<muladdsub_t>::operator();
    using ternary_broadcastable_operation<muladdsub_t>::operator();
};

template <>
struct operation_signature<muladdsub_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<muladdsub_t> : ternary_broadcasting_fallback<muladdsub_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, b, c);
        } else {
            return dx::addsub(dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a,
        simd_mask_type_t<AT> mask, BT b, CT c) noexcept(canonical_vector<AT> &&
        canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept(canonical_vector<AT> &&
        canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, M mask, BT b, CT c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero,
        simd_mask_type_t<AT> mask, AT a, BT b,
        CT c) noexcept(canonical_vector<AT> && canonical_vector<BT> &&
        canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero, M mask,
        AT a, BT b, CT c) noexcept(canonical_vector<AT> &&
        canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_mask<M> || extended_vector<AT> ||
        extended_vector<BT> || extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(a, b), c);
        }
    }

    using ternary_broadcasting_fallback<muladdsub_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::muladdsub_t muladdsub{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
