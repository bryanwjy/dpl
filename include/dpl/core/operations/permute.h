// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/extended_operations.h"
#include "dpl/core/operations/masked.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t...>
void permute(...) noexcept = delete;
void permute(...) noexcept = delete;

template <size_t... Is>
struct permutei_t;
struct permute_t;

template <typename T, typename Arg>
concept permute_result = simd_vector<T> && simd_vector<Arg> &&
    same_as<typename T::value_type, typename Arg::value_type> &&
    common_abi_with<typename T::abi_type, typename Arg::abi_type>;
template <typename T, typename Arg>
concept canonical_permute_result = permute_result<T, Arg> &&
    same_as<typename T::abi_type, typename Arg::abi_type>;

template <typename T, size_t... Is>
concept unqualified_canonical_permutei = requires(T val) {
    { permute<Is...>(internal::abi<T>, val) } -> canonical_permute_result<T>;
};

template <typename T, size_t... Is>
concept unqualified_extended_permutei = requires(T val) {
    { permute<Is...>(val) } -> permute_result<T>;
};

template <typename T, size_t... Is>
concept expression_permutei = simd_expression<T> &&
    invocable<permutei_t<Is...>, simd_expression_result_t<T>>;

template <typename T, size_t... Is>
concept decayable_permutei =
    decayable_vector_for<T, operation_category::lane_permutation> &&
    regular_invocable<permutei_t<Is...>, canonical_type_t<T>>;

template <typename T, size_t... Is>
concept extended_permutei = unqualified_extended_permutei<T, Is...> ||
    expression_permutei<T, Is...> || decayable_permutei<T, Is...>;

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mpermutei = requires(S src, M mask, T val) {
    {
        Op::native(internal::abi<A>, src, mask, val)
    } -> canonical_permute_result<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mpermutei = requires(S src, M mask, T val) {
    { Op::native(src, mask, val) } -> extended_operation_vector<A>;
};

template <typename Op, typename S, typename M, typename T>
concept expression_mpermutei =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    regular_invocable<Op, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>>;

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<T, M>>
concept decayable_mpermutei = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                  operation_category::lane_permutation> &&
    decayable_vector_for<T, operation_category::lane_permutation> &&
    decayable_mask_for<M, operation_category::lane_permutation> &&
    regular_invocable<Op, canonical_or_zero_t<S, T, A>, canonical_type_t<M>,
        canonical_type_t<T>>;

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<T, M>>
concept extended_mpermutei = unqualified_extended_mpermutei<Op, S, M, T, A> ||
    expression_mpermutei<Op, S, M, T> || decayable_mpermutei<Op, S, M, T, A>;

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_impermutei = requires(S src, M mask, T val) {
    {
        Op::native(internal::abi<A>, src,
            internal::to_const_mask<A, Op, S, T>(mask), val)
    } -> canonical_permute_result<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_impermutei = requires(S src, M mask, T val) {
    {
        Op::native(src, internal::to_const_mask<A, Op, S, T>(mask), val)
    } -> extended_operation_vector<A>;
};

template <typename Op, typename S, typename M, typename T>
concept expression_impermutei = (simd_expression<S> || simd_expression<T>) &&
    regular_invocable<Op, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>>;

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_impermutei = decayable_vector_for<canonical_if_zero_t<S, T>,
                                   operation_category::lane_permutation> &&
    decayable_vector_for<T, operation_category::lane_permutation> &&
    regular_invocable<Op, canonical_or_zero_t<S, T, A>, M, canonical_type_t<T>>;

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_impermutei = unqualified_extended_impermutei<Op, S, M, T, A> ||
    expression_impermutei<Op, S, M, T> || decayable_impermutei<Op, S, M, T, A>;

template <size_t... Is>
struct permutei_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        if constexpr (same_as<index_sequence<Is...>, iota_sequence_t<E, A>>) {
            return arg;
        } else {
            constexpr auto simd_size = simd_abi_traits<E, A>::size();
            return dx::initialize<E, A>(
                (Is < simd_size ? arg[imm<Is>] : E())...);
        }
    }

    template <typename T>
    static consteval auto extend() noexcept {
        return []<size_t... Js>(index_sequence<Js...>) {
            return permutei_t<Is..., (sizeof...(Is) + Js)...>{};
        }(iota_sequence<T>);
    }

    template <typename A, typename S, typename M, typename T>
    requires maskable_args<S, M, T> && requires(S src, M mask, T val) {
        permute<Is...>(internal::abi<A>, src, mask, val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(
        A abi, S src, M mask, T val) noexcept {
        return permute<Is...>(abi, src, mask, val);
    }

    template <typename A, typename M, typename T>
    requires zmaskable_args<M, T> && requires(M mask, T val) {
        permute<Is...>(internal::abi<A>, dx::zero, mask, val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(
        A abi, dx::zero_t tag, M mask, T val) noexcept {
        return permute<Is...>(abi, tag, mask, val);
    }

    template <typename A, typename S, typename M, typename T>
    requires imm_maskable_args<S, M, T> && requires(S src, M mask, T val) {
        permute<Is...>(internal::abi<A>, src,
            internal::to_const_mask<A, permutei_t, S, T>(mask), val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(
        A abi, S src, M mask, T val) noexcept {
        return permute<Is...>(abi, src, mask, val);
    }

    template <typename A, typename M, typename T>
    requires imm_zmaskable_args<M, T> && requires(M mask, T val) {
        permute<Is...>(internal::abi<A>, dx::zero,
            internal::to_const_mask<A, permutei_t, dx::zero_t, T>(mask), val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(
        A abi, dx::zero_t tag, M mask, T val) noexcept {
        return permute<Is...>(abi, tag, mask, val);
    }

    template <typename S, typename M, typename T>
    requires maskable_args<S, M, T> &&
        requires(S src, M mask, T val) { permute<Is...>(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(S src, M mask, T val) noexcept {
        return permute<Is...>(src, mask, val);
    }

    template <typename M, typename T>
    requires zmaskable_args<M, T> &&
        requires(M mask, T val) { permute<Is...>(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(
        dx::zero_t tag, M mask, T val) noexcept {
        return permute<Is...>(tag, mask, val);
    }

    template <typename S, typename M, typename T,
        typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
    requires imm_maskable_args<S, M, T> && requires(S src, M mask, T val) {
        permute<Is...>(
            src, internal::to_const_mask<A, permutei_t, S, T>(mask), val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(S src, M mask, T val) noexcept {
        return permute<Is...>(src, mask, val);
    }

    template <typename M, typename T, typename A = typename T::abi_type>
    requires imm_zmaskable_args<M, T> && requires(M mask, T val) {
        permute<Is...>(dx::zero,
            internal::to_const_mask<A, permutei_t, dx::zero_t, T>(mask), val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL native(
        dx::zero_t tag, M mask, T val) noexcept {
        return permute<Is...>(tag, mask, val);
    }

public:
    template <fixed_width_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept
    requires (sizeof...(Is) < simd_abi_traits<T>::size) &&
        regular_invocable<decltype(permutei_t::extend()), T>
    {
        constexpr auto permute = permutei_t::extend();
        return permute(arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires (sizeof...(Is) == simd_abi_traits<A, E>::size)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_permutei<basic_vector<E, A>,
                          Is...>) {
            if consteval {
                return fallback(arg);
            } else {
                return permute<Is...>(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <fixed_width_vector T>
    requires extended_vector<T> &&
        (sizeof...(Is) == simd_abi_traits<T>::size) &&
        extended_permutei<T, Is...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_permutei<T, Is...>) {
            return permute<Is...>(arg);
        } else if constexpr (expression_permutei<T, Is...>) {
            return operator()(dx::evaluate(arg));
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <fixed_width_vector S, fixed_width_mask M, fixed_width_vector T>
    requires maskable_args<T, M, S> &&
        (sizeof...(Is) < simd_abi_traits<T>::size) &&
        regular_invocable<decltype(permutei_t::extend()), S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto permute = permutei_t::extend();
        return permute(src, mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME> &&
        (sizeof...(Is) == simd_abi_traits<A, E>::size)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mpermutei<permutei_t,
                          basic_vector<E, A>, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<permutei_t>(src, mask, val);
            } else {
                return permute<Is...>(internal::abi<A>, src, mask, val);
            }
        } else {
            return internal::masked<permutei_t>(src, mask, val);
        }
    }

    template <fixed_width_abi MA, simd_element_for<MA> ME, fixed_width_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (sizeof...(Is) == simd_abi_traits<TA, E>::size) &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mpermutei<permutei_t, basic_vector<E, MA>,
            basic_mask<ME, MA>, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return permute<Is...>(internal::abi<MA>, src, mask, val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (sizeof...(Is) == simd_abi_traits<T>::size) &&
        (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mpermutei<permutei_t, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mpermutei<permutei_t, S, M, T>) {
            return permute<Is...>(src, mask, val);
        } else if constexpr (expression_mpermutei<permutei_t, S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME> &&
        (sizeof...(Is) == simd_abi_traits<A, E>::size)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mpermutei<permutei_t, dx::zero_t,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<permutei_t>(mask, val);
            } else {
                return permute<Is...>(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <fixed_width_abi MA, simd_element_for<MA> ME, fixed_width_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> && different_from<MA, TA> &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mpermutei<permutei_t, dx::zero_t,
            basic_mask<ME, MA>, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return permute<Is...>(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mpermutei<permutei_t, dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mpermutei<permutei_t, dx::zero_t, M,
                          T>) {
            return permute<Is...>(mask, val);
        } else if constexpr (expression_mpermutei<permutei_t, dx::zero_t, M,
                                 T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(val));
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<permutei_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_impermutei<permutei_t,
                          basic_vector<E, A>, M, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<permutei_t>(src, mask, val);
            } else {
                return permute<Is...>(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<permutei_t>(src, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires different_from<SA, TA> && simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_impermutei<permutei_t, basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> src, M mask, basic_vector<E, TA> val) noexcept {
        return permute<Is...>(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_impermutei<permutei_t, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_impermutei<permutei_t, S, M, T>) {
            return permute<Is...>(
                src, dx::to_compatible_const_mask<S>(mask), val);
        } else if constexpr (expression_impermutei<permutei_t, S, M, T>) {
            return operator()(dx::evaluate(src), mask, dx::evaluate(val));
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_impermutei<permutei_t, dx::zero_t,
                          M, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<permutei_t>(mask, val);
            } else {
                return permute<Is...>(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> &&
        extended_impermutei<permutei_t, dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_impermutei<permutei_t, dx::zero_t, M,
                          T>) {
            return permute<Is...>(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else if constexpr (expression_impermutei<permutei_t, dx::zero_t, M,
                                 T>) {
            return operator()(mask, dx::evaluate(val));
        } else {
            return operator()(mask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires invocable<permutei_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

struct permute_t;

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_canonical_permute = requires(T val, I idx) {
    { permute(internal::abi<A>, val, idx) } -> canonical_permute_result<T>;
};

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_extended_permute = requires(T val, I idx) {
    { permute(val, idx) } -> extended_operation_vector<A>;
};

template <typename T, typename I>
concept expression_permute =
    simd_expression<T> && invocable<permute_t, simd_expression_result_t<T>, I>;

template <typename T, typename I>
concept decayable_permute =
    decayable_vector_for<T, operation_category::lane_permutation> &&
    regular_invocable<permute_t, canonical_type_t<T>, I>;

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept extended_permute = unqualified_extended_permute<T, I, A> ||
    expression_permute<T, I> || decayable_permute<T, I>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mpermute = requires(S src, M mask, L lhs, R rhs) {
    {
        permute(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mpermute = requires(S src, M mask, L lhs, R rhs) {
    { permute(src, mask, lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mpermute =
    (simd_expression<S> || simd_expression<M> || simd_expression<L>) &&
    invocable<permute_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mpermute =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>,
        operation_category::lane_permutation> &&
    decayable_mask_for<M, operation_category::lane_permutation> &&
    decayable_vector_for<L, operation_category::lane_permutation> &&
    regular_invocable<permute_t,
        canonical_or_zero_t<S, operation_result_t<permute_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mpermute = unqualified_extended_mpermute<S, M, L, R, A> ||
    expression_mpermute<S, M, L, R> || decayable_mpermute<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept unqualified_canonical_impermute =
    requires(S src, M mask, L lhs, R rhs) {
        {
            permute(internal::abi<A>, src,
                internal::to_const_mask<A, permute_t, S, L, R>(mask), lhs, rhs)
        } -> equivalent_simd_as<
            canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>>;
    };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept unqualified_extended_impermute = requires(S src, M mask, L lhs, R rhs) {
    {
        permute(
            src, internal::to_const_mask<A, permute_t, S, L, R>(mask), lhs, rhs)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_impermute =
    (simd_expression<S> || simd_expression<M> || simd_expression<L>) &&
    regular_invocable<permute_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept decayable_impermute =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>,
        operation_category::lane_permutation> &&
    decayable_vector_for<L, operation_category::lane_permutation> &&
    decayable_vector_for<R, operation_category::lane_permutation> &&
    regular_invocable<permute_t,
        canonical_or_zero_t<S, operation_result_t<permute_t, L, R>, A>, M,
        canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept extended_impermute = unqualified_extended_impermute<S, M, L, R, A> ||
    expression_impermute<S, M, L, R> || decayable_impermute<S, M, L, R, A>;

struct permute_t {
private:
    template <canonical_class T, common_size_with<simd_lane_type_t<T>> E,
        same_as<typename T::abi_type> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallback(
        T arg, basic_vector<E, A> idx) noexcept {
        return []<size_t... Is>(
                   T arg, basic_vector<E, A> idx, index_sequence<Is...>) {
            constexpr auto simd_size = simd_abi_traits<T>::size();
            using TE = typename T::value_type;
            auto const zero = TE();
            return dx::initialize<T>(
                (idx[imm<Is>] < simd_size ? arg[idx[imm<Is>]] : zero)...);
        }(arg, idx, iota_sequence<T>);
    }

public:
    template <simd_vector T, size_t... Is>
    requires regular_invocable<permutei_t<Is...>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, index_sequence<Is...>) noexcept {
        constexpr permutei_t<Is...> permute{};
        return permute(val);
    }

    template <typename S, simd_mask M, simd_vector T, size_t... Is>
    requires regular_invocable<permutei_t<Is...>, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, M mask, T val, index_sequence<Is...>) noexcept {
        constexpr permutei_t<Is...> permute{};
        return permute(src, mask, val);
    }

    template <typename S, typename M, simd_vector T, size_t... Is>
    requires const_mask_for<M, T> &&
        regular_invocable<permutei_t<Is...>, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, M mask, T val, index_sequence<Is...>) noexcept {
        constexpr permutei_t<Is...> permute{};
        return permute(src, mask, val);
    }

    template <simd_mask M, simd_vector T, size_t... Is>
    requires regular_invocable<permutei_t<Is...>, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, T val, index_sequence<Is...>) noexcept {
        constexpr permutei_t<Is...> permute{};
        return permute(mask, val);
    }

    template <typename M, simd_vector T, size_t... Is>
    requires const_mask_for<M, T> && regular_invocable<permutei_t<Is...>, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, T val, index_sequence<Is...>) noexcept {
        constexpr permutei_t<Is...> permute{};
        return permute(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires common_size_with<E, I> && integral<I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> val, basic_vector<I, A> idx) noexcept {
        if constexpr (unqualified_canonical_permute<basic_vector<E, A>,
                          basic_vector<I, A>>) {
            if consteval {
                return fallback(val, idx);
            } else {
                return permute(internal::abi<A>, val, idx);
            }
        } else {
            return fallback(val, idx);
        }
    }

    template <simd_abi LA, simd_abi RA, simd_element_for<LA> E,
        simd_element_for<RA> I>
    requires common_size_with<E, I> && integral<I> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_permute<basic_vector<E, RA>, basic_vector<I, LA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, LA> val, basic_vector<I, RA> idx) noexcept {
        return permute(internal::abi<common_abi_t<LA, RA>>, val, idx);
    }

    template <extended_vector T, canonical_vector I>
    requires integral<typename I::value_type> && extended_permute<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I idx) noexcept {
        if constexpr (unqualified_extended_permute<T, I>) {
            return permute(val, idx);
        } else if constexpr (expression_permute<T, I>) {
            return operator()(dx::evaluate(val), idx);
        } else {
            return operator()(dx::to_canonical(val), idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_vector<LE, A> src,
        basic_mask<ME, A> mask, basic_vector<LE, A> lhs,
        basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mpermute<basic_vector<LE, A>,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(src, mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<permute_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> LE, common_size_with<LE> ME,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> &&
        (common_size_with<LE, RE> || different_from<LA, RA> ||
            scalable_abi<SA> || scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<LE, LA> &&
        maskable_args<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_mpermute<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<ME, SA> mask, basic_vector<LE, LA> lhs,
        basic_vector<RE, RA> rhs) noexcept {
        return permute(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, canonical_vector R>
    requires integral<typename R::value_type> &&
        (extended_vector<S> || extended_mask<M> || extended_vector<L>) &&
        maskable_args<S, M, L, R> && extended_mpermute<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mpermute<S, M, L, R>) {
            return permute(src, mask, lhs, rhs);
        } else if constexpr (expression_mpermute<S, M, L, R>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(lhs), rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_mask<ME, A> mask,
        basic_vector<LE, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mpermute<zero_t, basic_mask<ME, A>,
                          basic_vector<LE, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::broadcast<LE, A>(dx::zero), mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> &&
        (different_from<SA, common_abi_t<LA, RA>> || different_from<LA, RA> ||
            scalable_abi<SA> || scalable_abi<LA> || scalable_abi<RA> ||
            !common_size_with<E, RE>) &&
        simd_element_for<E, LA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<RE, RA>> &&
        unqualified_canonical_mpermute<zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        return permute(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, canonical_vector R>
    requires (extended_mask<M> || extended_vector<L>) &&
        integral<typename R::value_type> && zmaskable_args<M, L, R> &&
        extended_mpermute<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mpermute<zero_t, M, L, R>) {
            return permute(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_mpermute<zero_t, M, L, R>) {
            return operator()(dx::evaluate(mask), dx::evaluate(lhs), rhs);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(lhs), rhs);
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<permute_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> RE,
        const_mask_for<basic_vector<E, A>> M>
    requires integral<RE> && common_size_with<E, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_impermute<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(src, mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<permute_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi LA,
        common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> &&
        (different_from<SA, common_abi_t<LA, RA>> || different_from<LA, RA> ||
            scalable_abi<SA> || scalable_abi<LA> || scalable_abi<RA> ||
            !common_size_with<RE, E>) &&
        simd_element_for<E, LA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_impermute<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        return permute(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L,
        canonical_vector R>
    requires (extended_vector<S> || extended_vector<L>) &&
        integral<typename R::value_type> && imm_maskable_args<S, L, R> &&
        extended_impermute<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_impermute<S, M, L, R>) {
            return permute(
                src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_impermute<S, M, L, R>) {
            return operator()(
                dx::evaluate(src), mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> RE,
        const_mask_for<basic_vector<E, A>> M>
    requires integral<RE> && common_size_with<RE, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_impermute<zero_t, M,
                          basic_vector<E, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<permute_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E, simd_element_for<RA> RE,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires integral<RE> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
            !common_size_with<RE, E>) &&
        simd_element_for<E, LA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_impermute<zero_t, M, basic_vector<E, LA>,
            basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return permute(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <extended_vector L, canonical_vector R,
        const_mask_for<operation_result_t<permute_t, L, R>> M>
    requires imm_zmaskable_args<L, R> && extended_impermute<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<permute_t, L, R>;
        if constexpr (unqualified_extended_impermute<zero_t, M, L, R>) {
            return permute(
                dx::zero, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_impermute<zero_t, M, L, R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<permute_t, L, R>> M>
    requires invocable<permute_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <size_t... Is>
inline constexpr internal::permutei_t<Is...> permutei{};
DPL_EXPORT inline constexpr internal::permute_t permute{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
