// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwshift_right(...) noexcept = delete;
template <auto>
void bwshift_right(...) noexcept = delete;

struct bwshift_right_t;

template <typename T>
concept unqualified_canonical_bwsr = requires(T val, size_t shift) {
    {
        bwshift_right(internal::abi<T>, val, shift)
    } -> canonical_bitshift_result<T>;
};

template <typename T>
concept unqualified_extended_bwsr = requires(T val, size_t shift) {
    { bwshift_right(val, shift) } -> extended_bitshift_result<T>;
};

template <typename T>
concept unqualified_bwsr = unqualified_extended_bwsr<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<bwshift_right_t, canonical_type_t<T>, size_t>);

template <typename T>
concept unqualified_masksr = unqualified_extended_bwsr<T> ||
    (decayable_mask_for<T, operation_category::lane_permutation> &&
        regular_invocable<bwshift_right_t, canonical_type_t<T>, size_t>);

template <typename T, typename V>
concept unqualified_canonical_bwsri = requires(T val, V shift) {
    {
        bwshift_right(internal::abi<T>, val, shift)
    } -> canonical_bitshift_result<T>;
};

template <typename T, typename V>
concept unqualified_extended_bwsri = requires(T val, V shift) {
    { bwshift_right(val, shift) } -> extended_bitshift_result<T>;
};

template <typename T, typename V>
concept unqualified_bwsri = unqualified_extended_bwsri<T, V> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<bwshift_right_t, canonical_type_t<T>, V>);

template <typename T, typename V>
concept unqualified_masksri = unqualified_extended_bwsri<T, V> ||
    (decayable_mask_for<T, operation_category::lane_permutation> &&
        regular_invocable<bwshift_right_t, canonical_type_t<T>, V>);

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_bwsrv = requires(L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, lhs, rhs)
    } -> canonical_bitwise_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_bwsrv = requires(L lhs, R rhs) {
    { bwshift_right(lhs, rhs) } -> extended_bitwise_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwsrv = unqualified_canonical_bwsrv<L, R, A> ||
    unqualified_extended_bwsrv<L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<bwshift_right_t, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mbwsr =
    requires(S src, C mask, T val, size_t shift) {
        {
            bwshift_right(internal::abi<A>, src, mask, val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mbwsr =
    requires(S src, C mask, T val, size_t shift) {
        {
            bwshift_right(src, mask, val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept decayable_mbwsr = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                              operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(bwshift_right_t bwsr, canonical_or_zero_t<S, T, A> s,
        canonical_type_t<C> c, canonical_type_t<T> t) { bwsr(s, c, t); };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept extended_mbwsr =
    unqualified_extended_mbwsr<S, C, T, A> || decayable_mbwsr<S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbwsr =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_right(internal::abi<A>, src,
                internal::to_const_mask<A, bwshift_right_t, S, T, size_t>(mask),
                val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbwsr =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_right(src,
                internal::to_const_mask<A, bwshift_right_t, S, T, size_t>(mask),
                val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbwsr = decayable_vector_for<canonical_if_zero_t<S, T>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(bwshift_right_t bwsr, canonical_or_zero_t<S, T, A> src, M mask,
        canonical_type_t<T> val, size_t shift) { bwsr(src, mask, val, shift); };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbwsr =
    unqualified_extended_imbwsr<S, M, T, A> || decayable_imbwsr<S, M, T, A>;

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<C, L>>
concept unqualified_canonical_mbwsri = requires(S src, C mask, L val, R shift) {
    {
        bwshift_right(internal::abi<A>, src, mask, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<C, L>>
concept unqualified_extended_mbwsri = requires(S src, C mask, L val, R shift) {
    {
        bwshift_right(src, mask, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, C>>
concept decayable_mbwsri = decayable_vector_for<canonical_if_zero_t<S, L, A>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(bwshift_right_t bwsr, canonical_or_zero_t<S, L, A> src,
        canonical_type_t<C> mask, canonical_type_t<L> val,
        R shift) { bwsr(src, mask, val, shift); };

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, C>>
concept extended_mbwsri = unqualified_extended_mbwsri<S, C, L, R, A> ||
    decayable_mbwsri<S, C, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept unqualified_canonical_imbwsri =
    requires(S src, M mask, L val, R shift) {
        {
            bwshift_right(internal::abi<A>, src,
                internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), val,
                shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
    };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept unqualified_extended_imbwsri = requires(S src, M mask, L val, R shift) {
    {
        bwshift_right(src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), val,
            shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept decayable_imbwsri = decayable_vector_for<canonical_if_zero_t<S, L>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    requires(bwshift_right_t bwsr, canonical_or_zero_t<S, L, A> src, M mask,
        canonical_type_t<L> val, R shift) { bwsr(src, mask, val, shift); };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept extended_imbwsri = unqualified_extended_imbwsri<S, M, L, R, A> ||
    decayable_imbwsri<S, M, L, R, A>;

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_canonical_mbwsrv = requires(S src, C mask, L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_extended_mbwsrv = requires(S src, C mask, L lhs, R rhs) {
    {
        bwshift_right(src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept decayable_mbwsrv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(bwshift_right_t bwsr,
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A> s,
        canonical_type_t<C> c, canonical_type_t<L> l,
        canonical_type_t<R> r) { bwsr(s, c, l, r); };

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept extended_mbwsrv = unqualified_extended_mbwsrv<S, C, L, R, A> ||
    decayable_mbwsrv<S, C, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept unqualified_canonical_imbwsrv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), lhs,
            rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept unqualified_extended_imbwsrv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), lhs,
            rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept decayable_imbwsrv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(bwshift_right_t bwsr,
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A> s,
        M mask, canonical_type_t<L> l,
        canonical_type_t<R> r) { bwsr(s, mask, l, r); };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept extended_imbwsrv = unqualified_extended_imbwsrv<S, M, L, R, A> ||
    decayable_imbwsrv<S, M, L, R, A>;

struct bwshift_right_t {
private:
    template <typename LE, integral RE, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<LE, A> arg, basic_vector<RE, A> shift) noexcept {
        using bit_type = bit_type_t<sizeof(LE) * char_bit_v>;
        return internal::transform<basic_vector<LE, A>>(
            [](auto lhs, auto rhs) {
                if constexpr (integral<LE> && sizeof(LE) < sizeof(int)) {
                    return static_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) >> rhs);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) >> rhs);
                }
            },
            arg, shift);
    }

    template <typename L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_vector<L, A>>(
            [shift](auto lhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> shift);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> shift);
                }
            },
            arg);
    }

    template <typename L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_mask<L, A>;
        return []<size_t... Is>(
                   mask_type arg, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>(bitset(
                (Is + shift < simd_abi_traits<L, A>::size ? arg[Is + shift]
                                                          : false)...));
        }(arg, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_bwsr<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwsr<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_bwsr<basic_mask<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwsr<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <extended_vector T>
    requires unqualified_bwsr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_bwsr<T>) {
            return bwshift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <extended_mask T>
    requires unqualified_masksr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_bwsr<T>) {
            return bwshift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    //

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_bwsri<basic_vector<E, A>, Shift>) {
            if consteval {
                return operator()(val, Shift::value);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return operator()(val, Shift::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    requires unqualified_canonical_bwsri<basic_vector<E, A>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, Shift shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_bwsri<basic_mask<E, A>, Shift>) {
            if consteval {
                return operator()(val, Shift::value);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return operator()(val, Shift::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    requires unqualified_canonical_bwsri<basic_mask<E, A>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, Shift shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <extended_vector T, integral_constant_like Shift>
    requires unqualified_bwsri<T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Shift shift) noexcept {
        if constexpr (unqualified_extended_bwsri<T, Shift>) {
            return bwshift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <extended_mask T, integral_constant_like Shift>
    requires unqualified_masksri<T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Shift shift) noexcept {
        if constexpr (unqualified_extended_bwsri<T, Shift>) {
            return bwshift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    //

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE> && integral<RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(
        basic_vector<LE, A> val, basic_vector<RE, A> shift) noexcept {
        if constexpr (unqualified_canonical_bwsrv<basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <simd_abi LA, simd_element_for<LA> LE, common_abi_with<LA> RA,
        simd_element_for<RA> RE>
    requires (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        integral<RE> &&
        unqualified_canonical_bwsrv<basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, common_abi_t<LA, RA>> operator()(
        basic_vector<LE, LA> val, basic_vector<RE, RA> shift) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        common_abi_with<typename L::abi_type, typename R::abi_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        integral<typename R::value_type> && unqualified_bwsrv<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, R shift) noexcept {
        if constexpr (unqualified_extended_bwsrv<L, R>) {
            return bwshift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    ///
    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<ME, A> mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_mbwsr<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(
                    pass, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, pass, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwsl<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> pass,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val,
        size_t shift) noexcept {
        return bwshift_right(internal::abi<MA>, pass, mask, val, shift);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwsr<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S pass, M mask, T arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<S, M, T>) {
            return bwshift_right(pass, mask, arg, shift);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_mbwsr<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbwsr<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, MA> mask,
        basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<MA>, dx::zero, mask, val, shift);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwsr<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<dx::zero_t, M, T>) {
            return bwshift_right(mask, arg, shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(arg), shift);
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T arg, size_t shift) {
        bwshift_right_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, size_t shift) noexcept {
        return operator()(mask, arg, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        M mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_imbwsr<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(
                    pass, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi InA>
    requires (different_from<SA, InA> || scalable_abi<SA> ||
                 scalable_abi<InA>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, InA>> &&
        unqualified_canonical_imbwsr<basic_vector<E, SA>, M,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        M mask, basic_vector<E, InA> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val,
            shift);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwsr<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S pass, M mask, T arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<S, M, T>) {
            return bwshift_right(
                pass, dx::to_compatible_const_mask<S>(mask), arg, shift);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg),
                shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_imbwsr<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> M>
    requires scalable_abi<InA> && simd_element_for<E, InA> &&
        imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imbwsr<zero_t, M, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        M mask, basic_vector<E, InA> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<InA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, InA>>(mask), val,
            shift);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbwsr<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<zero_t, M, T>) {
            return bwshift_right(
                dx::zero, dx::to_compatible_const_mask<T>(mask), arg, shift);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires requires(M mask, T arg, size_t shift) {
        bwshift_right_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, size_t shift) noexcept {
        return operator()(mask, arg, shift);
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<ME, A> mask, basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_mbwsri<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_right_t>(
                    pass, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, pass, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like Shift>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwsri<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> pass,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val,
        Shift shift) noexcept {
        return bwshift_right(internal::abi<MA>, pass, mask, val, shift);
    }

    template <simd_vector S, simd_mask M, simd_vector T,
        integral_constant_like Shift>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwsri<S, M, T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S pass, M mask, T arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<S, M, T, Shift>) {
            return bwshift_right(pass, mask, arg, shift);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_mbwsri<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like Shift>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbwsri<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, MA> mask,
        basic_vector<E, TA> val, Shift shift) noexcept {
        return bwshift_right(internal::abi<MA>, dx::zero, mask, val, shift);
    }

    template <simd_mask M, simd_vector T, integral_constant_like Shift>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwsri<zero_t, M, T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<dx::zero_t, M, T, Shift>) {
            return bwshift_right(mask, arg, shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(arg), shift);
        }
    }

    template <simd_mask M, simd_vector T, integral_constant_like Shift>
    requires requires(M mask, T arg, Shift shift) {
        bwshift_right_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, Shift shift) noexcept {
        return operator()(mask, arg, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        M mask, basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_imbwsri<basic_vector<E, A>, M,
                          basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_right_t>(
                    pass, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi InA,
        integral_constant_like Shift>
    requires (different_from<SA, InA> || scalable_abi<SA> ||
                 scalable_abi<InA>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, InA>> &&
        unqualified_canonical_imbwsri<basic_vector<E, SA>, M,
            basic_vector<E, InA>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        M mask, basic_vector<E, InA> val, Shift shift) noexcept {
        return bwshift_right(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val,
            shift);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        integral_constant_like Shift>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwsri<S, M, T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S pass, M mask, T arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<S, M, T, Shift>) {
            return bwshift_right(
                pass, dx::to_compatible_const_mask<S>(mask), arg, shift);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg),
                shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_imbwsri<zero_t, M,
                          basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> M, integral_constant_like Shift>
    requires scalable_abi<InA> && simd_element_for<E, InA> &&
        imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imbwsri<zero_t, M, basic_vector<E, InA>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        M mask, basic_vector<E, InA> val, Shift shift) noexcept {
        return bwshift_right(internal::abi<InA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, InA>>(mask), val,
            shift);
    }

    template <extended_vector T, const_mask_for<T> M,
        integral_constant_like Shift>
    requires imm_zmaskable_args<T> && extended_imbwsri<zero_t, M, T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<zero_t, M, T, Shift>) {
            return bwshift_right(
                dx::zero, dx::to_compatible_const_mask<T>(mask), arg, shift);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <simd_vector T, const_mask_for<T> M, integral_constant_like Shift>
    requires requires(M mask, T arg, Shift shift) {
        bwshift_right_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, Shift shift) noexcept {
        return operator()(mask, arg, shift);
    }
    ///
    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_vector<LE, A> src,
        basic_mask<ME, A> mask, basic_vector<LE, A> lhs,
        basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwsrv<bwshift_right_t,
                          basic_vector<LE, A>, basic_mask<ME, A>,
                          basic_vector<LE, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_right(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> LE, common_size_with<LE> ME,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> && common_size_with<LE, RE> &&
        (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
            scalable_abi<RA>) &&
        simd_element_for<LE, LA> &&
        maskable_args<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_mbwsrv<bwshift_right_t, basic_vector<LE, SA>,
            basic_mask<ME, SA>, basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<ME, SA> mask, basic_vector<LE, LA> lhs,
        basic_vector<RE, RA> rhs) noexcept {
        return bwshift_right(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires integral<typename R::value_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
            extended_vector<R>) &&
        maskable_args<S, M, L, R> &&
        extended_mbwsrv<bwshift_right_t, S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwsrv<bwshift_right_t, S, M, L,
                          R>) {
            return bwshift_right(src, mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_mask<ME, A> mask,
        basic_vector<LE, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwsrv<bwshift_right_t, zero_t,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, lhs, rhs);
            } else {
                return bwshift_right(
                    internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::broadcast<LE, A>(dx::zero), mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mbwsrv<bwshift_right_t, zero_t,
            basic_mask<ME, SA>, basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_right(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> &&
        extended_mbwsrv<bwshift_right_t, zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwsrv<bwshift_right_t, zero_t, M, L,
                          R>) {
            return bwshift_right(mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires requires(
        M mask, L lhs, R rhs) { bwshift_right_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imbwsrv<bwshift_right_t,
                          basic_vector<E, A>, M, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_right(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi LA,
        common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_imbwsrv<bwshift_right_t, basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_right(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> &&
        extended_imbwsrv<bwshift_right_t, S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imbwsrv<bwshift_right_t, S, M, L,
                          R>) {
            return bwshift_right(
                src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imbwsrv<bwshift_right_t, zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, lhs, rhs);
            } else {
                return bwshift_right(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imbwsrv<bwshift_right_t, zero_t, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_right(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwshift_right_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> &&
        extended_imbwsrv<bwshift_right_t, zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<bwshift_right_t, L, R>;
        if constexpr (unqualified_extended_imbwsrv<bwshift_right_t, zero_t, M,
                          L, R>) {
            return bwshift_right(
                dx::zero, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R, const_mask_like M>
    requires requires(
        M mask, L lhs, R rhs) { bwshift_right_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

template <size_t N>
struct bwshift_righti_t {
    template <typename... Args>
    requires invocable<bwshift_right_t, Args..., size_constant<N>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args... val) noexcept {
        constexpr size_constant<N> shift;
        return bwshift_right_t::operator()(val..., shift);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwshift_right_t bwshift_right{};
DPL_EXPORT template <size_t N>
inline constexpr internal::bwshift_righti_t<N> bwshift_righti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
