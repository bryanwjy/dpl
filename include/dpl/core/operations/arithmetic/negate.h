// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/arithmetic/subtract.h"
#include "dpl/core/operations/masked.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/zero.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void negate(...) noexcept = delete;
template <auto>
void negate(...) noexcept = delete;

struct negate_t;

template <typename T>
concept unqualified_canonical_negate = requires(T val) {
    {
        negate(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_negate = requires(T val) {
    { negate(val) } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_negate = unqualified_extended_negate<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<negate_t, canonical_type_t<T>>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mnegate = requires(S src, C mask, T val) {
    {
        negate(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mnegate = requires(S src, C mask, T val) {
    {
        negate(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept decayable_mnegate = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, canonical_type_t<C> c,
        canonical_type_t<T> t) { Op::operator()(s, c, t); };

template <typename Op, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept extended_mnegate = unqualified_extended_mnegate<S, C, T, A> ||
    decayable_mnegate<Op, S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imnegate = requires(S src, M mask, T val) {
    {
        negate(internal::abi<A>, src,
            internal::to_const_mask<A, negate_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imnegate = requires(S src, M mask, T val) {
    {
        negate(src, internal::to_const_mask<A, negate_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imnegate = decayable_vector_for<canonical_if_zero_t<S, T>,
                                 operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, M mask, canonical_type_t<T> t) {
        Op::operator()(s, mask, t);
    };

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imnegate = unqualified_extended_imnegate<S, M, T, A> ||
    decayable_imnegate<Op, S, M, T, A>;

struct negate_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> val) noexcept {
        return dx::subtract(dx::zero, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_negate<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return negate(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !arithmetic_type<E>) &&
        unqualified_canonical_negate<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return negate(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_negate<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_negate<T>) {
            return negate(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mnegate<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(pass, mask, val);
            } else {
                return negate(internal::abi<A>, pass, mask, val);
            }
        } else {
            return internal::masked<negate_t>(pass, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, A2> &&
        maskable_args<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>> &&
        unqualified_canonical_mnegate<basic_vector<E, A1>,
            basic_mask<MaskE, A1>, basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A1> operator()(basic_vector<E, A1> pass,
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return negate(internal::abi<A1>, pass, mask, val);
    }

    template <simd_vector S, simd_mask Mask, simd_vector Arg>
    requires (extended_vector<S> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<S, Mask, Arg> && extended_mnegate<negate_t, S, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<S, Mask, Arg>) {
            return negate(pass, mask, arg);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mnegate<zero_t,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                return negate(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, A2> &&
        zmaskable_args<basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mnegate<dx::zero_t, basic_mask<MaskE, A1>,
            basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return negate(internal::abi<A1>, dx::zero, mask, val);
    }

    template <simd_mask Mask, simd_vector Arg>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_mnegate<negate_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<dx::zero_t, Mask, Arg>) {
            return negate(mask, arg);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(arg));
        }
    }

    template <simd_mask Mask, simd_vector Arg>
    requires requires(Mask mask, Arg arg) { negate_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> pass, Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imnegate<basic_vector<E, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(pass, mask, val);
            } else {
                return negate(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<negate_t>(pass, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi InA>
    requires (different_from<SA, InA> || scalable_abi<SA> ||
                 scalable_abi<InA> || !arithmetic_type<E>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, InA>> &&
        unqualified_canonical_imnegate<basic_vector<E, SA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        Mask mask, basic_vector<E, InA> val) noexcept {
        return negate(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector Arg>
    requires (extended_vector<S> || extended_vector<Arg>) &&
        imm_maskable_args<S, Arg> && extended_imnegate<negate_t, S, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<S, Mask, Arg>) {
            return negate(pass, dx::to_compatible_const_mask<S>(mask), arg);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imnegate<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                return negate(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask>
    requires (scalable_abi<InA> || !arithmetic_type<E>) &&
        simd_element_for<E, InA> && imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imnegate<zero_t, Mask, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        Mask mask, basic_vector<E, InA> val) noexcept {
        return negate(internal::abi<InA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, InA>>(mask), val);
    }

    template <extended_vector Arg, const_mask_for<Arg> Mask>
    requires imm_zmaskable_args<Arg> &&
        extended_imnegate<negate_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_imnegate<zero_t, Mask, Arg>) {
            return negate(
                dx::zero, dx::to_compatible_const_mask<Arg>(mask), arg);
        } else {
            return operator()(
                dx::to_compatible_const_mask<Arg>(mask), dx::to_canonical(arg));
        }
    }

    template <simd_vector Arg, const_mask_for<Arg> Mask>
    requires requires(Mask mask, Arg arg) { negate_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::negate_t negate{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
