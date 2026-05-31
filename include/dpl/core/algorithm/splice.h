// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void splice(...) noexcept = delete;

struct splice_t;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_splice = requires(S src, M mask, T val) {
    {
        splice(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_splice = requires(S src, M mask, T val) {
    {
        splice(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<T, M>>
concept decayable_splice = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, canonical_type_t<M> c,
        canonical_type_t<T> t) { Op::operator()(s, c, t); };

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<T, M>>
concept extended_splice =
    unqualified_extended_splice<S, M, T, A> || decayable_splice<Op, S, M, T, A>;

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_canonical_splicei = requires(S src, R val) {
    {
        splice(internal::abi<A>, src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_extended_splicei = requires(S src, R val) {
    {
        splice(src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<M, R>>
concept unqualified_splicei = unqualified_extended_splice<S, M, R, A> ||
    (decayable_vector_for<S, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        regular_invocable<splice_t, canonical_type_t<S>, M,
            canonical_type_t<R>>);

struct splice_t {
private:
    template <typename EM, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> lhs,
        basic_mask<EM, A> mask, basic_vector<E, A> rhs) noexcept {
        using mask_type = basic_mask<EM, A>;
        auto const low = dx::countr_zero(mask);
        auto const high = dx::countl_zero(mask);
        return dx::slide_left(dx::shift_right(lhs, high), rhs, high + low);
    }

    template <auto V, typename EL, typename ER, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<EL, A> lhs, basic_vector<ER, A> rhs) noexcept {
        constexpr const_mask<simd_abi_traits<EL, A>::size, V> inmask{};
        constexpr auto low = dx::countr_zero(inmask);
        constexpr auto high = dx::countr_zero(inmask);
        return dx::slide_left(
            dx::shift_right(lhs, imm<high>), rhs, imm<high + low>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_splice<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return fallback(src, mask, val);
            } else {
                return splice(internal::abi<A>, src, mask, val);
            }
        } else {
            return fallback(src, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> && different_from<MA, TA> &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_splice<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return splice(internal::abi<MA>, src, mask, val);
    }

    template <simd_vector S, simd_mask Mask, simd_vector Arg>
    requires (extended_vector<S> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<S, Mask, Arg> && extended_splice<splice_t, S, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_splice<S, Mask, Arg>) {
            return splice(src, mask, arg);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_splice<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
            } else {
                return splice(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E, common_abi_with<TA> MA,
        simd_element_for<MA> ME>
    requires common_size_with<E, ME> && different_from<MA, TA> &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_splice<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return splice(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask Mask, simd_vector Arg>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_splice<splice_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_splice<dx::zero_t, Mask, Arg>) {
            return splice(mask, arg);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(arg));
        }
    }

    template <simd_mask Mask, simd_vector Arg>
    requires requires(Mask mask, Arg arg) { splice_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, Mask mask, basic_vector<E, A> val) noexcept {
        constexpr auto cmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
        if constexpr (unqualified_canonical_splicei<basic_vector<E, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                constexpr auto V = cmask();
                return splice_t::template fallbacki<V>(src, mask, val);
            } else {
                return splice(internal::abi<A>, src, cmask, val);
            }
        } else {
            constexpr auto V = cmask();
            return splice_t::template fallbacki<V>(src, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_splicei<basic_vector<E, SA>, Mask,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> src, Mask mask, basic_vector<E, TA> val) noexcept {
        return splice(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector Arg>
    requires (extended_vector<S> || extended_vector<Arg>) &&
        imm_maskable_args<S, Arg> && extended_splicei<splice_t, S, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_splice<S, Mask, Arg>) {
            return splice(src, dx::to_compatible_const_mask<S>(mask), arg);
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_splicei<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
            } else {
                constexpr auto cmask =
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
                return splice(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires (scalable_abi<TA> || !arithmetic_type<E>) &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_splicei<zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val) noexcept {
        return splice(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_splicei<splice_t, zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_splicei<zero_t, M, T>) {
            return splice(dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else {
            return operator()(mask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires requires(M mask, T val) { splice_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

template <auto V>
struct splicei_t {};

template <integral auto V>
struct splicei_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <typename S, simd_vector R>
    requires requires {
        typename canonical_if_zero_t<S, R>;
        typename mask_type<canonical_if_zero_t<S, R>>;
        requires regular_invocable<splice_t, S,
            mask_type<canonical_if_zero_t<S, R>>, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<canonical_if_zero_t<S, R>> mask{};
        return splice_t::operator()(src, mask, val);
    }

    template <simd_vector T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<splice_t, dx::zero_t, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        constexpr mask_type<T> mask{};
        return splice_t::operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
DPL_EXPORT inline constexpr internal::splice_t splice{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
