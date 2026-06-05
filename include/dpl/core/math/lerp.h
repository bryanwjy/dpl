// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic/result.h"
#  include "dpl/core/operations/internal/masked.h"
#  include "dpl/core/operations/internal/operation_base.h"
#  include "dpl/core/operations/internal/transform.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void lerp(...) noexcept = delete;
template <auto>
void lerp(...) noexcept = delete;

struct lerp_t;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_lerp = requires(AT a, BT b, CT c) {
    { lerp(internal::abi<A>, a, b, c) } -> canonical_fma_result<AT, BT, CT, A>;
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_lerp = requires(AT a, BT b, CT c) {
    { lerp(a, b, c) } -> extended_operation_vector<A>;
};

template <typename AT, typename BT, typename CT>
concept expression_lerp =
    (simd_expression<AT> || simd_expression<BT> || simd_expression<CT>) &&
    invocable<lerp_t, simd_expression_result_t<AT>,
        simd_expression_result_t<BT>, simd_expression_result_t<CT>>;

template <typename AT, typename BT, typename CT>
concept decayable_lerp =
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<lerp_t, canonical_type_t<AT>, canonical_type_t<BT>,
        canonical_type_t<CT>>;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept extended_lerp = unqualified_extended_lerp<AT, BT, CT, A> ||
    expression_lerp<AT, BT, CT> || decayable_lerp<AT, BT, CT>;

struct lerp_t :
    private ternary_operation_base<lerp_t>,
    private mx::masked_operation<lerp_t> {
private:
    friend ternary_operation_base<lerp_t>;
    friend mx::masked_operation<lerp_t>;

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((canonical_vector<AT> || !simd_vector<AT>) &&
        (canonical_vector<BT> || !simd_vector<BT>) &&
        (canonical_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        {
            lerp(internal::abi<A>, a, b, c)
        } -> broadcasting_arithmetic_result<A>;
    }
    {
        return lerp(internal::abi<A>, a, b, c);
    }

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((extended_vector<AT> || !simd_vector<AT>) ||
        (extended_vector<BT> || !simd_vector<BT>) ||
        (extended_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        { lerp(a, b, c) } -> broadcasting_arithmetic_result<A>;
    }
    {
        return lerp(a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> start,
        basic_vector<E, A> end, basic_vector<E, A> scale) noexcept {
        return dx::fmadd(scale, dx::subtract(end, start), start);
    }

    template <simd_vector S, typename M, simd_vector AT, simd_vector BT,
        simd_vector CT>
    requires mx::maskable_operator<lerp_t, S, M, AT, BT, CT> &&
        mx::canonical_operator_args<S, M, AT, BT, CT> &&
        requires(S src, M mask, AT a, BT b, CT c) {
            lerp(internal::abi<common_abi_t<AT, BT, CT>>, src, mask, a, b, c);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, AT a, BT b, CT c) noexcept {
        return lerp(
            internal::abi<common_abi_t<AT, BT, CT>>, src, mask, a, b, c);
    }

    template <simd_vector S, typename M, simd_vector AT, simd_vector BT,
        simd_vector CT>
    requires mx::maskable_operator<lerp_t, S, M, AT, BT, CT> &&
        (!mx::canonical_operator_args<S, M, AT, BT, CT>) &&
        requires(S src, M mask, AT a, BT b, CT c) { lerp(src, mask, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, AT a, BT b, CT c) noexcept {
        return lerp(src, mask, a, b, c);
    }

    template <typename M, simd_vector AT, simd_vector BT, simd_vector CT>
    requires mx::maskable_zoperator<lerp_t, M, AT, BT, CT> &&
        mx::canonical_zoperator_args<lerp_t, M, AT, BT, CT> &&
        requires(M mask, AT a, BT b, CT c) {
            lerp(internal::abi<common_abi_t<AT, BT, CT>>, dx::zero, mask, a, b,
                c);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, AT a, BT b, CT c) noexcept {
        return lerp(
            internal::abi<common_abi_t<AT, BT, CT>>, dx::zero, mask, a, b, c);
    }

    template <typename M, simd_vector AT, simd_vector BT, simd_vector CT>
    requires mx::maskable_zoperator<lerp_t, M, AT, BT, CT> &&
        (!mx::canonical_zoperator_args<lerp_t, M, AT, BT, CT>) &&
        requires(M mask, AT a, BT b, CT c) { lerp(dx::zero, mask, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, AT a, BT b, CT c) noexcept {
        return lerp(dx::zero, mask, a, b, c);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_lerp<basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return lerp(internal::abi<A>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA,
        typename E>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !floating_point<E>) &&
        unqualified_canonical_lerp<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return lerp(internal::abi<common_abi_t<AA, BA, CA>>, a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        extended_lerp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_lerp<AT, BT, CT>) {
            return lerp(a, b, c);
        } else if constexpr (expression_lerp<AT, BT, CT>) {
            return operator()(
                dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using ternary_operation_base<lerp_t>::operator();
    using mx::masked_operation<lerp_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::lerp_t lerp{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
