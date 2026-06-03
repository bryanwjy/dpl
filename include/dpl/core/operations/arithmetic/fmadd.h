// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/multiply.h"
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/extended_operations.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void fmadd(...) noexcept = delete;

struct fmadd_t;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_fmadd = requires(AT a, BT b, CT c) {
    { fmadd(internal::abi<A>, a, b, c) } -> canonical_fma_result<AT, BT, CT, A>;
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_fmadd = requires(AT a, BT b, CT c) {
    { fmadd(a, b, c) } -> extended_operation_vector<A>;
};

template <typename AT, typename BT, typename CT>
concept expression_fmadd =
    (simd_expression<AT> || simd_expression<BT> || simd_expression<CT>) &&
    invocable<fmadd_t, simd_expression_result_t<AT>,
        simd_expression_result_t<BT>, simd_expression_result_t<CT>>;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept decayable_fmadd =
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<fmadd_t, canonical_type_t<AT>, canonical_type_t<BT>,
        canonical_type_t<CT>>;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_fmadd = unqualified_extended_fmadd<AT, BT, CT, A> ||
    expression_fmadd<AT, BT, CT> || decayable_fmadd<AT, BT, CT, A>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept unqualified_canonical_mfmadd =
    requires(S src, M mask, AT a, BT b, CT c) {
        {
            fmadd(internal::abi<A>, src, mask, a, b, c)
        } -> equivalent_simd_as<
            canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>, A>>;
    };

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept unqualified_extended_mfmadd =
    requires(S src, M mask, AT a, BT b, CT c) {
        { fmadd(src, mask, a, b, c) } -> extended_operation_vector<A>;
    };
template <typename S, typename M, typename AT, typename BT, typename CT>
concept expression_mfmadd =
    (simd_expression<S> || simd_expression<M> || simd_expression<AT> ||
        simd_expression<BT> || simd_expression<CT>) &&
    invocable<fmadd_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<AT>,
        simd_expression_result_t<BT>, simd_expression_result_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept decayable_mfmadd =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<fmadd_t,
        canonical_or_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>, A>,
        canonical_type_t<M>, canonical_type_t<AT>, canonical_type_t<BT>,
        canonical_type_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept extended_mfmadd = unqualified_extended_mfmadd<S, M, AT, BT, CT, A> ||
    expression_mfmadd<S, M, AT, BT, CT> ||
    decayable_mfmadd<S, M, AT, BT, CT, A>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>>,
        operation_result_t<fmadd_t, AT, BT, CT>>>
concept unqualified_canonical_imfmadd = requires(
    S src, M mask, AT a, BT b, CT c) {
    {
        fmadd(internal::abi<A>, src,
            internal::to_const_mask<A, fmadd_t, S, AT, BT, CT>(mask), a, b, c)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>, A>>;
};

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>>,
        operation_result_t<fmadd_t, AT, BT, CT>>>
concept unqualified_extended_imfmadd =
    requires(S src, M mask, AT a, BT b, CT c) {
        {
            fmadd(src, internal::to_const_mask<A, fmadd_t, S, AT, BT, CT>(mask),
                a, b, c)
        } -> extended_operation_vector<A>;
    };

template <typename S, typename M, typename AT, typename BT, typename CT>
concept expression_imfmadd = (simd_expression<S> || simd_expression<AT> ||
                                 simd_expression<BT> || simd_expression<CT>) &&
    invocable<fmadd_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<AT>, simd_expression_result_t<BT>,
        simd_expression_result_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>>,
        operation_result_t<fmadd_t, AT, BT, CT>>>
concept decayable_imfmadd =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<fmadd_t,
        canonical_or_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>, A>, M,
        canonical_type_t<AT>, canonical_type_t<BT>, canonical_type_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmadd_t, AT, BT, CT>>,
        operation_result_t<fmadd_t, AT, BT, CT>>>
concept extended_imfmadd = unqualified_extended_imfmadd<S, M, AT, BT, CT, A> ||
    expression_imfmadd<S, M, AT, BT, CT> ||
    decayable_imfmadd<S, M, AT, BT, CT, A>;

struct fmadd_t : private ternary_operation_base<fmadd_t> {
private:
    friend ternary_operation_base<fmadd_t>;

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((canonical_vector<AT> || !simd_vector<AT>) &&
        (canonical_vector<BT> || !simd_vector<BT>) &&
        (canonical_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        {
            fmadd(internal::abi<A>, a, b, c)
        } -> broadcasting_arithmetic_result<A>;
    }
    {
        return fmadd(internal::abi<A>, a, b, c);
    }

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((extended_vector<AT> || !simd_vector<AT>) ||
        (extended_vector<BT> || !simd_vector<BT>) ||
        (extended_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        { fmadd(a, b, c) } -> extended_operation_vector<A>;
    }
    {
        return fmadd(a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return dx::add(dx::multiply(a, b), c);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_fmadd<basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fmadd(internal::abi<A>, a, b, c);
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
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        unqualified_canonical_fmadd<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmadd(internal::abi<common_abi_t<AA, BA, CA>>, a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_fmadd<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_fmadd<AT, BT, CT>) {
            return fmadd(a, b, c);
        } else if constexpr (expression_fmadd<AT, BT, CT>) {
            return operator()(
                dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using ternary_operation_base<fmadd_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_mfmadd<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmadd_t>(src, mask, a, b, c);
            } else {
                return fmadd(internal::abi<A>, src, mask, a, b, c);
            }
        } else {
            return internal::masked<fmadd_t>(src, mask, a, b, c);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        maskable_args<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_mfmadd<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<ME, SA> mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmadd(internal::abi<SA>, src, mask, a, b, c);
    }

    template <simd_vector S, simd_mask M, simd_vector AT, simd_vector BT,
        simd_vector CT>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        maskable_args<S, M, AT, BT, CT> && extended_mfmadd<S, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_mfmadd<S, M, AT, BT, CT>) {
            return fmadd(src, mask, a, b, c);
        } else if constexpr (expression_mfmadd<S, M, AT, BT, CT>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_mfmadd<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmadd_t>(mask, a, b, c);
            } else {
                return fmadd(internal::abi<A>, dx::zero, mask, a, b, c);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, a, b, c);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_mfmadd<zero_t, basic_mask<ME, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmadd(
            internal::abi<common_abi_t<AA, BA, CA>>, dx::zero, mask, a, b, c);
    }

    template <simd_mask M, simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        zmaskable_args<M, AT, BT, CT> && extended_mfmadd<zero_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_mfmadd<zero_t, M, AT, BT, CT>) {
            return fmadd(dx::zero, mask, a, b, c);
        } else if constexpr (expression_mfmadd<zero_t, M, AT, BT, CT>) {
            return operator()(dx::evaluate(mask), dx::evaluate(a),
                dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <simd_mask M, simd_vector AT, simd_vector BT, simd_vector CT>
    requires requires(
        M mask, AT a, BT b, CT c) { fmadd_t::operator()(mask, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, AT a, BT b, CT c) noexcept {
        return operator()(mask, a, b, c);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_imfmadd<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmadd_t>(src, mask, a, b, c);
            } else {
                return fmadd(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), a,
                    b, c);
            }
        } else {
            return internal::masked<fmadd_t>(src, mask, a, b, c);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi AA,
        common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_imfmadd<basic_vector<E, SA>, M,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmadd(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), a, b, c);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector AT,
        simd_vector BT, simd_vector CT>
    requires (extended_vector<S> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        imm_maskable_args<S, AT, BT, CT> && extended_imfmadd<S, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_imfmadd<S, M, AT, BT, CT>) {
            return fmadd(src, dx::to_compatible_const_mask<S>(mask), a, b, c);
        } else if constexpr (expression_imfmadd<S, M, AT, BT, CT>) {
            return operator()(dx::evaluate(src), mask, dx::evaluate(a),
                dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_canonical(src), mask, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(M mask, basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_imfmadd<zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmadd_t>(mask, a, b, c);
            } else {
                return fmadd(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), a,
                    b, c);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, a, b, c);
        }
    }

    template <typename M, simd_abi AA, common_abi_with<AA> BA,
        common_abi_with<BA> CA, simd_element_for<AA> E>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, BA> && simd_element_for<E, CA> &&
        const_mask_for<M, basic_vector<E, common_abi_t<AA, BA, CA>>> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        imm_zmaskable_args<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>> &&
        unqualified_canonical_imfmadd<zero_t, M, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        M mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        using A = common_abi_t<AA, BA, CA>;
        return fmadd(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT,
        const_mask_for<operation_result_t<fmadd_t, AT, BT, CT>> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        imm_zmaskable_args<AT, BT, CT> &&
        extended_imfmadd<zero_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, AT a, BT b, CT c) noexcept {
        using S = operation_result_t<fmadd_t, AT, BT, CT>;
        if constexpr (unqualified_extended_imfmadd<zero_t, M, AT, BT, CT>) {
            return fmadd(
                dx::zero, dx::to_compatible_const_mask<S>(mask), a, b, c);
        } else if constexpr (expression_imfmadd<zero_t, M, AT, BT, CT>) {
            return operator()(
                mask, dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT, const_mask_like M>
    requires requires(
        M mask, AT a, BT b, CT c) { fmadd_t::operator()(mask, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, AT a, BT b, CT c) noexcept {
        return operator()(mask, a, b, c);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::fmadd_t fmadd{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
