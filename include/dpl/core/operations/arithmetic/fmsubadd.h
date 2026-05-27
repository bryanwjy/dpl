// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/fmadd.h"
#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/zero.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void fmsubadd(...) noexcept = delete;
template <auto>
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
    { fmsubadd(a, b, c) } -> extended_fma_result<AT, BT, CT, A>;
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_fmsubadd = unqualified_extended_fmsubadd<AT, BT, CT, A> ||
    (decayable_vector_for<AT, operation_category::lane_agnostic> &&
        decayable_vector_for<BT, operation_category::lane_agnostic> &&
        decayable_vector_for<CT, operation_category::lane_agnostic> &&
        regular_invocable<fmsubadd_t, canonical_type_t<AT>,
            canonical_type_t<BT>, canonical_type_t<CT>>);

template <typename S, typename C, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, C>>
concept unqualified_canonical_mfmsubadd = requires(
    S src, C mask, AT a, BT b, CT c) {
    {
        fmsubadd(internal::abi<A>, src, mask, a, b, c)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>>;
};

template <typename S, typename C, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, C>>
concept unqualified_extended_mfmsubadd = requires(
    S src, C mask, AT a, BT b, CT c) {
    {
        fmsubadd(src, mask, a, b, c)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>>;
};

template <typename S, typename C, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, C>>
concept decayable_mfmsubadd =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    decayable_vector_for<AT, operation_category::lane_agnostic> &&
    decayable_vector_for<BT, operation_category::lane_agnostic> &&
    decayable_vector_for<CT, operation_category::lane_agnostic> &&
    requires(fmsubadd_t op,
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A> s,
        canonical_type_t<C> mask, canonical_type_t<AT> a,
        canonical_type_t<BT> b,
        canonical_type_t<CT> c) { op(s, mask, a, b, c); };

template <typename S, typename C, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT, C>>
concept extended_mfmsubadd =
    unqualified_extended_mfmsubadd<S, C, AT, BT, CT, A> ||
    decayable_mfmsubadd<S, C, AT, BT, CT, A>;

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
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>>;
};

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>>,
        operation_result_t<fmsubadd_t, AT, BT, CT>>>
concept unqualified_extended_imfmsubadd = requires(
    S src, M mask, AT a, BT b, CT c) {
    {
        fmsubadd(src,
            internal::to_const_mask<A, fmsubadd_t, S, AT, BT, CT>(mask), a, b,
            c)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A>>;
};

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
    requires(fmsubadd_t op,
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>, A> s,
        M mask, canonical_type_t<AT> a, canonical_type_t<BT> b,
        canonical_type_t<CT> c) { op(s, mask, a, b, c); };

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<fmsubadd_t, AT, BT, CT>>,
        operation_result_t<fmsubadd_t, AT, BT, CT>>>
concept extended_imfmsubadd =
    unqualified_extended_imfmsubadd<S, M, AT, BT, CT, A> ||
    decayable_imfmsubadd<S, M, AT, BT, CT, A>;

struct fmsubadd_t : ternary_operation_base<fmsubadd_t> {
private:
    friend ternary_operation_base<fmsubadd_t>;

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((canonical_vector<AT> || !simd_vector<AT>) &&
        (canonical_vector<BT> || !simd_vector<BT>) &&
        (canonical_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        {
            fmsubadd(internal::abi<A>, a, b, c)
        } -> broadcasting_arithmetic_result<A>;
    }
    {
        return fmsubadd(internal::abi<A>, a, b, c);
    }

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((extended_vector<AT> || !simd_vector<AT>) ||
        (extended_vector<BT> || !simd_vector<BT>) ||
        (extended_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        { fmsubadd(a, b, c) } -> broadcasting_arithmetic_result<A>;
    }
    {
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
    requires arithmetic_type<E>
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
            different_from<BA, CA> || !arithmetic_type<E>) &&
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
        } else {
            return operator()(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using ternary_operation_base<fmsubadd_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<MaskE, A> mask, basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_mfmsubadd<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>,
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

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        maskable_args<basic_vector<E, SA>, basic_mask<MaskE, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_mfmsubadd<basic_vector<E, SA>,
            basic_mask<MaskE, SA>, basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<MaskE, SA> mask, basic_vector<E, AA> a,
        basic_vector<E, BA> b, basic_vector<E, CA> c) noexcept {
        return fmsubadd(internal::abi<SA>, src, mask, a, b, c);
    }

    template <simd_vector S, simd_mask Mask, simd_vector AT, simd_vector BT,
        simd_vector CT>
    requires (extended_vector<S> || extended_mask<Mask> ||
                 extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        maskable_args<S, Mask, AT, BT, CT> &&
        extended_mfmsubadd<S, Mask, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, Mask mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_mfmsubadd<S, Mask, AT, BT, CT>) {
            return fmsubadd(src, mask, a, b, c);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_mfmsubadd<zero_t,
                          basic_mask<MaskE, A>, basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<fmsubadd_t>(mask, a, b, c);
            } else {
                return fmsubadd(internal::abi<A>, dx::zero, mask, a, b, c);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, a, b, c);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        zmaskable_args<basic_mask<MaskE, SA>, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_mfmsubadd<zero_t, basic_mask<MaskE, SA>,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, SA> mask,
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmsubadd(
            internal::abi<common_abi_t<AA, BA, CA>>, dx::zero, mask, a, b, c);
    }

    template <simd_mask Mask, simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_mask<Mask> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        zmaskable_args<Mask, AT, BT, CT> &&
        extended_mfmsubadd<zero_t, Mask, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_mfmsubadd<zero_t, Mask, AT, BT,
                          CT>) {
            return fmsubadd(mask, a, b, c);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <simd_mask Mask, simd_vector AT, simd_vector BT, simd_vector CT>
    requires requires(
        Mask mask, AT a, BT b, CT c) { fmsubadd_t::operator()(mask, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, AT a, BT b, CT c) noexcept {
        return operator()(mask, a, b, c);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        Mask mask, basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_imfmsubadd<basic_vector<E, A>, Mask,
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
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi AA,
        common_abi_with<AA> BA, common_abi_with<BA> CA>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>> &&
        unqualified_canonical_imfmsubadd<basic_vector<E, SA>, Mask,
            basic_vector<E, AA>, basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        Mask mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return fmsubadd(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), a, b, c);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector AT,
        simd_vector BT, simd_vector CT>
    requires (extended_vector<S> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        imm_maskable_args<S, AT, BT, CT> &&
        extended_imfmsubadd<S, Mask, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, Mask mask, AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_imfmsubadd<S, Mask, AT, BT, CT>) {
            return fmsubadd(
                src, dx::to_compatible_const_mask<S>(mask), a, b, c);
        } else {
            return operator()(dx::to_canonical(src), mask, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(Mask mask,
        basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_imfmsubadd<zero_t, Mask,
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

    template <typename Mask, simd_abi AA, common_abi_with<AA> BA,
        common_abi_with<BA> CA, simd_element_for<AA> E>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, BA> && simd_element_for<E, CA> &&
        const_mask_for<Mask, basic_vector<E, common_abi_t<AA, BA, CA>>> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !arithmetic_type<E>) &&
        imm_zmaskable_args<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>> &&
        unqualified_canonical_imfmsubadd<zero_t, Mask, basic_vector<E, AA>,
            basic_vector<E, BA>, basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        Mask mask, basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        using A = common_abi_t<AA, BA, CA>;
        return fmsubadd(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT,
        const_mask_for<operation_result_t<fmsubadd_t, AT, BT, CT>> Mask>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        imm_zmaskable_args<AT, BT, CT> &&
        extended_imfmsubadd<zero_t, Mask, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, AT a, BT b, CT c) noexcept {
        using S = operation_result_t<fmsubadd_t, AT, BT, CT>;
        if constexpr (unqualified_extended_imfmsubadd<zero_t, Mask, AT, BT,
                          CT>) {
            return fmsubadd(
                dx::zero, dx::to_compatible_const_mask<S>(mask), a, b, c);
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT,
        const_mask_like Mask>
    requires requires(
        Mask mask, AT a, BT b, CT c) { fmsubadd_t::operator()(mask, a, b, c); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, AT a, BT b, CT c) noexcept {
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
