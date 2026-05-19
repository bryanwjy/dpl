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
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void negate(...) noexcept = delete;
template <auto>
void negate(...) noexcept = delete;

template <typename T>
concept unqualified_canonical_negate = requires(T val) {
    {
        negate(internal::abi<T>, val)
    } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

template <typename T>
concept unqualified_extended_negate = requires(T val) {
    { negate(val) } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

template <typename T>
concept unqualified_negate =
    unqualified_canonical_negate<T> || unqualified_extended_negate<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        unqualified_canonical_negate<canonical_type_t<T>>);

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

template <typename Abs, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept decayable_mnegate = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, canonical_type_t<C> c,
        canonical_type_t<T> t) { Abs::operator()(s, c, t); };

template <typename Abs, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept extended_mnegate = unqualified_extended_mnegate<S, C, T, A> ||
    decayable_mnegate<Abs, S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_negatei = requires(S src, T val) {
    {
        negate<const_mask_v<canonical_if_zero_t<S, T>, M>>(
            internal::abi<A>, src, dx::masked_operation, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_negatei = requires(S src, T val) {
    {
        negate<const_mask_v<canonical_if_zero_t<S, T>, M>>(
            src, dx::masked_operation, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Abs, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_negatei = decayable_vector_for<canonical_if_zero_t<S, T>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, M mask, canonical_type_t<T> t) {
        Abs::operator()(s, mask, t);
    };

template <typename Abs, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_negatei = unqualified_extended_negatei<S, M, T, A> ||
    decayable_negatei<Abs, S, M, T, A>;

struct negate_t {
private:
    template <arithmetic_type E>
    using result DPL_NODEBUG = common_arithmetic_type_t<E, E>;

    template <arithmetic_vector T>
    using result_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> val) noexcept {
        if constexpr (unsigned_integral<E>) {
            return val;
        } else {
            return dx::subtract(dx::zero, val);
        }
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

    template <simd_vector Pass, simd_mask Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<Pass, Mask, Arg> &&
        extended_mnegate<negate_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Pass pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<Pass, Mask, Arg>) {
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
        if constexpr (unqualified_canonical_negatei<basic_vector<E, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(pass, mask, val);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return negate<V>(internal::abi<A>, pass, masked_operation, val);
            }
        } else {
            return internal::masked<negate_t>(pass, mask, val);
        }
    }

    template <simd_abi PassA, simd_element_for<PassA> E,
        const_mask_for<basic_vector<E, PassA>> Mask, simd_abi InA>
    requires (different_from<PassA, InA> || scalable_abi<PassA> ||
                 scalable_abi<InA> || !arithmetic_type<E>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, PassA>, basic_vector<E, InA>> &&
        unqualified_canonical_negatei<basic_vector<E, PassA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, PassA> operator()(
        basic_vector<E, PassA> pass, Mask mask,
        basic_vector<E, InA> val) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, PassA>, Mask>;
        return negate<V>(internal::abi<PassA>, pass, masked_operation, val);
    }

    template <simd_vector Pass, const_mask_for<Pass> Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_vector<Arg>) &&
        imm_maskable_args<Pass, Arg> &&
        extended_negatei<negate_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Pass pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<Pass, Mask, Arg>) {
            constexpr auto V = const_mask_v<Pass, Mask>;
            return negate<V>(pass, masked_operation, arg);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<Pass>(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_negatei<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return negate<V>(
                    internal::abi<A>, dx::zero, masked_operation, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask>
    requires (scalable_abi<InA> || !arithmetic_type<E>) &&
        simd_element_for<E, InA> && imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_negatei<zero_t, Mask, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        Mask mask, basic_vector<E, InA> val) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, InA>, Mask>;
        return negate<V>(internal::abi<InA>, dx::zero, masked_operation, val);
    }

    template <extended_vector Arg, const_mask_for<Arg> Mask>
    requires imm_zmaskable_args<Arg> &&
        extended_negatei<negate_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<zero_t, Mask, Arg>) {
            constexpr auto V = const_mask_v<Arg, Mask>;
            return negate<V>(dx::zero, masked_operation, arg);
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
