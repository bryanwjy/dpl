// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/fmadd.h"
#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/operation_base.h"
#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void fmsubadd(...) noexcept = delete;

struct fmsubadd_t;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_fmsubadd = requires(AT a, BT b, CT c) {
    {
        fmsubadd(internal::abi<A>, a, b, c)
    } -> canonical_fma_result<AT, BT, CT, A>;
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_fmsubadd = requires(AT a, BT b, CT c) {
    { fmsubadd(a, b, c) } -> vector_with_common_abi<A>;
};

template <typename AT, typename BT, typename CT>
concept expression_fmsubadd =
    (simd_expression<AT> || simd_expression<BT> || simd_expression<CT>) &&
    invocable<fmsubadd_t, simd_expression_result_t<AT>,
        simd_expression_result_t<BT>, simd_expression_result_t<CT>>;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept decayable_fmsubadd =
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<fmsubadd_t, canonical_type_t<AT>, canonical_type_t<BT>,
        canonical_type_t<CT>>;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_fmsubadd = unqualified_extended_fmsubadd<AT, BT, CT, A> ||
    expression_fmsubadd<AT, BT, CT> || decayable_fmsubadd<AT, BT, CT, A>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept unqualified_canonical_mfmsubadd = requires(
    S src, M mask, AT a, BT b, CT c) {
    {
        fmsubadd(internal::abi<A>, src, mask, a, b, c)
    } -> equivalent_simd_type_with<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>>;
};

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept unqualified_extended_mfmsubadd =
    requires(S src, M mask, AT a, BT b, CT c) {
        { fmsubadd(src, mask, a, b, c) } -> vector_with_common_abi<A>;
    };
template <typename S, typename M, typename AT, typename BT, typename CT>
concept expression_mfmsubadd =
    (simd_expression<S> || simd_expression<M> || simd_expression<AT> ||
        simd_expression<BT> || simd_expression<CT>) &&
    invocable<fmsubadd_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<AT>,
        simd_expression_result_t<BT>, simd_expression_result_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept decayable_mfmsubadd =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<fmsubadd_t,
        canonical_or_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>,
        canonical_type_t<M>, canonical_type_t<AT>, canonical_type_t<BT>,
        canonical_type_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, M>>
concept extended_mfmsubadd =
    unqualified_extended_mfmsubadd<S, M, AT, BT, CT, A> ||
    expression_mfmsubadd<S, M, AT, BT, CT> ||
    decayable_mfmsubadd<S, M, AT, BT, CT, A>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>>,
        operation_result_t<fmsubadd_t, AT, BT, CT>>>
concept unqualified_canonical_imfmsubadd = requires(
    S src, M mask, AT a, BT b, CT c) {
    {
        fmsubadd(internal::abi<A>, src,
            internal::to_const_mask<A, fmsubadd_t, S, AT, BT, CT>(mask), a, b,
            c)
    } -> equivalent_simd_type_with<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>>;
};

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>>,
        operation_result_t<fmsubadd_t, AT, BT, CT>>>
concept unqualified_extended_imfmsubadd =
    requires(S src, M mask, AT a, BT b, CT c) {
        {
            fmsubadd(src,
                internal::to_const_mask<A, fmsubadd_t, S, AT, BT, CT>(mask), a,
                b, c)
        } -> vector_with_common_abi<A>;
    };

template <typename S, typename M, typename AT, typename BT, typename CT>
concept expression_imfmsubadd =
    (simd_expression<S> || simd_expression<AT> || simd_expression<BT> ||
        simd_expression<CT>) &&
    invocable<fmsubadd_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<AT>, simd_expression_result_t<BT>,
        simd_expression_result_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>>,
        operation_result_t<fmsubadd_t, AT, BT, CT>>>
concept decayable_imfmsubadd =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    regular_invocable<fmsubadd_t,
        canonical_or_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>,
        M, canonical_type_t<AT>, canonical_type_t<BT>, canonical_type_t<CT>>;

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>>,
        operation_result_t<fmsubadd_t, AT, BT, CT>>>
concept extended_imfmsubadd =
    unqualified_extended_imfmsubadd<S, M, AT, BT, CT, A> ||
    expression_imfmsubadd<S, M, AT, BT, CT> ||
    decayable_imfmsubadd<S, M, AT, BT, CT, A>;

struct fmsubadd_t : private ternary_operation_base<fmsubadd_t> {
private:
    friend ternary_operation_base<fmsubadd_t>;

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires (!simd_type<AT> || canonical_vector<AT>) &&
        (!simd_type<BT> || canonical_vector<BT>) &&
        (!simd_type<CT> || canonical_vector<CT>) && requires(AT a, BT b, CT c) {
            { fmsubadd(internal::abi<A>, a, b, c) } -> vector_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept {
        return fmsubadd(internal::abi<A>, a, b, c);
    }

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires (!simd_type<AT> || extended_vector<AT>) &&
        (!simd_type<BT> || extended_vector<BT>) &&
        (!simd_type<CT> || extended_vector<CT>) &&
        unqualified_fmsubadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, AT a, BT b, CT c) noexcept(
        noexcept(fmsubadd(a, b, c))) {
        return fmsubadd(a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return dx::fmadd(a, b, dx::negate(c, imm<0b0101>, c));
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires basic_element<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_fmsubadd<basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return fmsubadd(internal::abi<A>, a, b, c);
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
            different_from<BA, CA> || !basic_element<E>) &&
        unqualified_canonical_fmsubadd<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmsubadd(internal::abi<common_abi_t<AA, BA, CA>>, a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_fmsubadd<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_fmsubadd<AT, BT, CT>) {
            return fmsubadd(a, b, c);
        } else if constexpr (expression_fmsubadd<AT, BT, CT>) {
            return operator()(
                dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using ternary_operation_base<fmsubadd_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires basic_element<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_mfmsubadd<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmsubadd_t>(src, mask, a, b, c);
            } else {
                return fmsubadd(internal::abi<A>, src, mask, a, b, c);
            }
        } else {
            return internal::masked<fmsubadd_t>(src, mask, a, b, c);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !basic_element<E>) &&
        maskable_args<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_mfmsubadd<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<ME, SA> mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmsubadd(internal::abi<SA>, src, mask, a, b, c);
    }

    template <simd_vector S, simd_mask M, simd_vector AT, simd_vector BT,
        simd_vector CT>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        maskable_args<S, M, AT, BT, CT> && extended_mfmsubadd<S, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_mfmsubadd<S, M, AT, BT, CT>) {
            return fmsubadd(src, mask, a, b, c);
        } else if constexpr (expression_mfmsubadd<S, M, AT, BT, CT>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires basic_element<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_mfmsubadd<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmsubadd_t>(mask, a, b, c);
            } else {
                return fmsubadd(internal::abi<A>, dx::zero, mask, a, b, c);
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
            different_from<BA, CA> || !basic_element<E>) &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_mfmsubadd<zero_t, basic_mask<ME, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmsubadd(
            internal::abi<common_abi_t<AA, BA, CA>>, dx::zero, mask, a, b, c);
    }

    template <simd_mask M, simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        zmaskable_args<M, AT, BT, CT> &&
        extended_mfmsubadd<zero_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_mfmsubadd<zero_t, M, AT, BT, CT>) {
            return fmsubadd(dx::zero, mask, a, b, c);
        } else if constexpr (expression_mfmsubadd<zero_t, M, AT, BT, CT>) {
            return operator()(dx::evaluate(mask), dx::evaluate(a),
                dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <simd_mask M, simd_vector AT, simd_vector BT, simd_vector CT>
    requires invocable<fmsubadd_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, AT a, BT b, CT c) noexcept {
        return operator()(mask, a, b, c);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires basic_element<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_imfmsubadd<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmsubadd_t>(src, mask, a, b, c);
            } else {
                return fmsubadd(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), a,
                    b, c);
            }
        } else {
            return internal::masked<fmsubadd_t>(src, mask, a, b, c);
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
            different_from<BA, CA> || !basic_element<E>) &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_imfmsubadd<basic_vector<E, SA>, M,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmsubadd(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), a, b, c);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector AT,
        simd_vector BT, simd_vector CT>
    requires (extended_vector<S> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        imm_maskable_args<S, AT, BT, CT> &&
        extended_imfmsubadd<S, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_imfmsubadd<S, M, AT, BT, CT>) {
            return fmsubadd(
                src, dx::to_compatible_const_mask<S>(mask), a, b, c);
        } else if constexpr (expression_imfmsubadd<S, M, AT, BT, CT>) {
            return operator()(dx::evaluate(src), mask, dx::evaluate(a),
                dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_canonical(src), mask, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires basic_element<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(M mask, basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_imfmsubadd<zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmsubadd_t>(mask, a, b, c);
            } else {
                return fmsubadd(internal::abi<A>, dx::zero,
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
            different_from<BA, CA> || !basic_element<E>) &&
        imm_zmaskable_args<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>> &&
        unqualified_canonical_imfmsubadd<zero_t, M, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        M mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        using A = common_abi_t<AA, BA, CA>;
        return fmsubadd(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT,
        const_mask_for<operation_result_t<fmsubadd_t, AT, BT, CT>> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        imm_zmaskable_args<AT, BT, CT> &&
        extended_imfmsubadd<zero_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, AT a, BT b, CT c) noexcept {
        using S = operation_result_t<fmsubadd_t, AT, BT, CT>;
        if constexpr (unqualified_extended_imfmsubadd<zero_t, M, AT, BT, CT>) {
            return fmsubadd(
                dx::zero, dx::to_compatible_const_mask<S>(mask), a, b, c);
        } else if constexpr (expression_imfmsubadd<zero_t, M, AT, BT, CT>) {
            return operator()(
                mask, dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT,
        const_mask_for<operation_result_t<fmsubadd_t, AT, BT, CT>> M>
    requires invocable<fmsubadd_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, AT a, BT b, CT c) noexcept {
        return operator()(mask, a, b, c);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::fmsubadd_t fmsubadd{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
