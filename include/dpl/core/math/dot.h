// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/evaluate.h"
#  include "dpl/core/operations/extended_operations.h"
#  include "dpl/core/operations/lane_index.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Computes the bfloat16 dot product with float32 accumulation.
 *
 * Performs a paired bfloat16 dot product, internally promoting each bfloat16
 * operand to float32 before multiplication and accumulation. The result is a
 * vector of float32 values, where each float32 element accumulates the dot
 * product of two consecutive bfloat16 pairs from the input vectors.
 *
 * @par Mathematical Definition
 * For each float32 lane j in the destination register:
 * @code
 * FOR j := 0 to N
 *     dst.fp32[j] += make_fp32(a.bf16[2*j+1]) * make_fp32(b.bf16[2*j+1])
 *     dst.fp32[j] += make_fp32(a.bf16[2*j+0]) * make_fp32(b.bf16[2*j+0])
 * ENDFOR
 * @endcode
 * where N is the number of float32 lanes in the vector register.
 */
void dot_product(...) noexcept = delete;
struct dot_product_t;

template <typename S, typename L, typename R,
    typename A = common_abi_t<L, R, S>>
concept unqualified_canonical_dp = requires(S acc, L left, R right) {
    {
        dot_product(internal::abi<A>, acc, left, right)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename L, typename R,
    typename A = common_abi_t<L, R, S>>
concept unqualified_extended_dp = requires(S acc, L left, R right) {
    { dot_product(acc, left, right) } -> extended_operation_vector<A>;
};

template <typename S, typename L, typename R>
concept expression_dp =
    (simd_expression<S> || simd_expression<L> || simd_expression<R>) &&
    __DPL invocable<dot_product_t, simd_expression_result_t<S>,
        simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename S, typename L, typename R>
concept decayable_dp =
    decayable_vector_for<S, operation_category::lane_reduction> &&
    decayable_vector_for<L, operation_category::lane_reduction> &&
    decayable_vector_for<R, operation_category::lane_reduction> &&
    unqualified_canonical_dp<canonical_type_t<S>, canonical_type_t<L>,
        canonical_type_t<R>>;

template <typename S, typename L, typename R,
    typename A = common_abi_t<L, R, S>>
concept extended_dp = unqualified_extended_dp<S, L, R, A> &&
    expression_dp<S, L, R> && decayable_dp<S, L, R>;

struct dot_product_t : private mx::masked_assignment<dot_product_t> {
private:
    friend mx::masked_assignment<dot_product_t>;

    template <brain_float E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A>
        DPL_VECTORCALL fallback(basic_vector<float, A> src,
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        auto const idx = dx::lane_index<E, A>();
        auto const lower_half = idx < basic_vector<float, A>::size();
        auto const even = dx::bwshift_left(lower_half, idx, imm<1zu>);
        auto const odd = dx::add(lower_half, even,
            dx::broadcast<decltype(dx::lane_index<E, A>())>(dx::one));
        auto const odd_vals =
            dx::multiply(dx::element_cast<float>(dx::permute(lhs, odd)),
                dx::element_cast<float>(dx::permute(rhs, odd)));
        auto const even_vals =
            dx::multiply(dx::element_cast<float>(dx::permute(lhs, even)),
                dx::element_cast<float>(dx::permute(rhs, even)));
        return dx::add(src, dx::add(odd_vals, even_vals));
    }

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires mx::canonical_masked_math_assignment<dot_product_t, S, M, L, R> &&
        requires(S src, M mask, L lhs, R rhs) {
            dot_product(internal::abi<common_abi_t<S, R>>, src, mask, lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<S, R>>, src, mask, lhs, rhs);
    }

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires mx::extended_masked_math_assignment<dot_product_t, S, M, L, R> &&
        requires(
            S src, M mask, L lhs, R rhs) { dot_product(src, mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, L lhs, R rhs) noexcept {
        return dot_product(src, mask, lhs, rhs);
    }

    template <typename M, simd_vector S, simd_vector L, simd_vector R>
    requires mx::canonical_masked_math_zassignment<dot_product_t, M, S, L, R> &&
        requires(M mask, S src, L lhs, R rhs) {
            dot_product(internal::abi<common_abi_t<S, R>>, dx::zero, mask, src,
                lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, S src, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<S, R>>, dx::zero, mask, src, lhs, rhs);
    }

    template <typename M, simd_vector S, simd_vector L, simd_vector R>
    requires mx::extended_masked_math_zassignment<dot_product_t, M, S, L, R> &&
        requires(M mask, S src, L lhs, R rhs) {
            dot_product(dx::zero, mask, src, lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, S src, L lhs, R rhs) noexcept {
        return dot_product(dx::zero, mask, src, lhs, rhs);
    }

public:
    template <brain_float E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<float, A> src,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_dp<basic_vector<float, A>,
                          basic_vector<E, A>, basic_vector<E, A>, A>) {
            if consteval {
                return fallback(src, lhs, rhs);
            } else {
                return dot_product(internal::abi<A>, src, lhs, rhs);
            }
        } else {
            return fallback(src, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> SE, simd_abi LA,
        simd_element_for<LA> LE, simd_abi RA, simd_element_for<RA> RE>
    requires (different_from<SA, LA> || different_from<LA, RA> ||
                 different_from<SA, RA> || different_from<LE, RE> ||
                 !brain_float<LE> || !brain_float<RE> ||
                 different_from<SE, float>) &&
        unqualified_canonical_dp<basic_vector<SE, SA>, basic_vector<LE, LA>,
            basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<SE, SA> src,
        basic_vector<LE, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA, SA>;
        return dot_product(internal::abi<A>, src, lhs, rhs);
    }

    template <simd_vector S, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        extended_dp<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_dp<S, L, R>) {
            return dot_product(src, lhs, rhs);
        } else if constexpr (expression_dp<S, L, R>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    using mx::masked_assignment<dot_product_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::dot_product_t dot_product{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
