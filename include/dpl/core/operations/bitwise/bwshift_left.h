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
void bwshift_left(...) noexcept = delete;
template <auto>
void bwshift_left(...) noexcept = delete;

template <typename T>
concept unqualified_canonical_bwsl = requires(T val, size_t shift) {
    { bwshift_left(internal::abi<T>, val, shift) } -> canonical_shift_result<T>;
};

template <typename T>
concept unqualified_extended_bwsl = requires(T val, size_t shift) {
    { bwshift_left(val, shift) } -> extended_shift_result<T>;
};

template <typename T>
concept unqualified_bwsl =
    unqualified_canonical_bwsl<T> || unqualified_extended_bwsl<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        unqualified_canonical_bwsl<canonical_type_t<T>>);

template <typename T>
concept unqualified_masksl =
    unqualified_canonical_bwsl<T> || unqualified_extended_bwsl<T> ||
    (decayable_mask_for<T, operation_category::lane_permutation> &&
        unqualified_canonical_bwsl<canonical_type_t<T>>);

template <typename T, typename V>
concept unqualified_canonical_bwsli = requires(T val, V shift) {
    { bwshift_left(internal::abi<T>, val, shift) } -> canonical_shift_result<T>;
};

template <typename T, typename V>
concept unqualified_extended_bwsli = requires(T val, V shift) {
    { bwshift_left(val, shift) } -> extended_shift_result<T>;
};

template <typename T, typename V>
concept unqualified_bwsli =
    unqualified_canonical_bwsli<T, V> || unqualified_extended_bwsli<T, V> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        unqualified_canonical_bwsli<canonical_type_t<T>, V>);

template <typename T, typename V>
concept unqualified_masksli =
    unqualified_canonical_bwsli<T, V> || unqualified_extended_bwsli<T, V> ||
    (decayable_mask_for<T, operation_category::lane_permutation> &&
        unqualified_canonical_bwsli<canonical_type_t<T>, V>);

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_bwslv = requires(L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, lhs, rhs)
    } -> canonical_bitwise_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_bwslv = requires(L lhs, R rhs) {
    { bwshift_left(lhs, rhs) } -> extended_bitwise_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_bwslv =
    unqualified_canonical_bwslv<L, R> || unqualified_extended_bwslv<L, R> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        unqualified_canonical_bwslv<canonical_type_t<L>, canonical_type_t<R>>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mbwsl =
    requires(S src, C mask, T val, size_t shift) {
        {
            bwshift_left(internal::abi<A>, src, mask, val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mbwsl =
    requires(S src, C mask, T val, size_t shift) {
        {
            bwshift_left(src, mask, val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename Op, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept decayable_mbwsl = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                              operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, canonical_type_t<C> c,
        canonical_type_t<T> t) { Op::operator()(s, c, t); };

template <typename Op, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept extended_mbwsl =
    unqualified_extended_mbwsl<S, C, T, A> || decayable_mbwsl<Op, S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbwsl = requires(S src, T val, size_t shift) {
    {
        bwshift_left<const_mask_v<canonical_if_zero_t<S, T>, M>>(
            internal::abi<A>, src, dx::masked_operation, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbwsl = requires(S src, T val, size_t shift) {
    {
        bwshift_left<const_mask_v<canonical_if_zero_t<S, T>, M>>(
            src, dx::masked_operation, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbwsl = decayable_vector_for<canonical_if_zero_t<S, T>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> src, M mask, canonical_type_t<T> val,
        size_t shift) { Op::operator()(src, mask, val, shift); };

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbwsl =
    unqualified_extended_imbwsl<S, M, T, A> || decayable_imbwsl<Op, S, M, T, A>;

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<C, L>>
concept unqualified_canonical_mbwsli = requires(S src, C mask, L val, R shift) {
    {
        bwshift_left(internal::abi<A>, src, mask, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<C, L>>
concept unqualified_extended_mbwsli = requires(S src, C mask, L val, R shift) {
    {
        bwshift_left(src, mask, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, C>>
concept decayable_mbwsli = decayable_vector_for<canonical_if_zero_t<S, L, A>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, L, A> src, canonical_type_t<C> mask,
        canonical_type_t<L> val,
        R shift) { Op::operator()(src, mask, val, shift); };

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, C>>
concept extended_mbwsli = unqualified_extended_mbwsli<S, C, L, R, A> ||
    decayable_mbwsli<Op, S, C, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept unqualified_canonical_imbwsli = requires(S src, L val, R shift) {
    {
        bwshift_left<const_mask_v<canonical_if_zero_t<S, L>, M>>(
            internal::abi<A>, src, dx::masked_operation, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept unqualified_extended_imbwsli = requires(S src, L val, R shift) {
    {
        bwshift_left<const_mask_v<canonical_if_zero_t<S, L>, M>>(
            src, dx::masked_operation, val, shift)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename Op, typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept decayable_imbwsli = decayable_vector_for<canonical_if_zero_t<S, L>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, L, A> src, M mask, canonical_type_t<L> val,
        R shift) { Op::operator()(src, mask, val, shift); };

template <typename Op, typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept extended_imbwsli = unqualified_extended_imbwsli<S, M, L, R, A> ||
    decayable_imbwsli<Op, S, M, L, R, A>;

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_canonical_mbwslv = requires(S src, C mask, L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_extended_mbwslv = requires(S src, C mask, L lhs, R rhs) {
    {
        bwshift_left(src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept decayable_mbwslv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(canonical_if_zero_t<S, operation_result_t<Op, L, R>, A> s,
        canonical_type_t<C> c, canonical_type_t<L> l,
        canonical_type_t<R> r) { Op::operator()(s, c, l, r); };

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept extended_mbwslv = unqualified_extended_mbwslv<Op, S, C, L, R, A> ||
    decayable_mbwslv<Op, S, C, L, R, A>;

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept unqualified_canonical_imbwslv = requires(S src, L lhs, R rhs) {
    {
        bwshift_left<const_mask_v<
            canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>, M>>(
            internal::abi<A>, src, dx::masked_operation, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept unqualified_extended_imbwslv = requires(S src, L lhs, R rhs) {
    {
        bwshift_left<const_mask_v<
            canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>, M>>(
            src, dx::masked_operation, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept decayable_imbwslv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(canonical_if_zero_t<S, operation_result_t<Op, L, R>, A> s, M mask,
        canonical_type_t<L> l,
        canonical_type_t<R> r) { Op::operator()(s, mask, l, r); };

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept extended_imbwslv = unqualified_extended_imbwslv<Op, S, M, L, R, A> ||
    decayable_imbwslv<Op, S, M, L, R, A>;

struct bwshift_left_t {
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
                        __DPL bit_cast<bit_type>(lhs) << rhs);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << rhs);
                }
            },
            arg, shift);
    }

    template <typename LE, simd_abi LA, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<LE, LA> arg, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(LE) * char_bit_v>;
        return internal::transform<basic_vector<LE, LA>>(
            [](auto lhs) {
                if constexpr (integral<LE> && sizeof(LE) < sizeof(int)) {
                    return static_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << R::value);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << R::value);
                }
            },
            arg);
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
                        __DPL bit_cast<bit_type>(lhs) << shift);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << shift);
                }
            },
            arg);
    }

    template <typename LE, simd_abi LA, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<LE, LA> arg, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(LE) * char_bit_v>;
        using mask_type = basic_mask<LE, LA>;
        if constexpr (R::value >= mask_type::size()) {
            return dx::broadcast<LE, LA>(false_type{});
        } else {
            return []<size_t... Is>(mask_type arg, index_sequence<Is...>) {
                return dx::initialize<mask_type>(
                    bitset((Is >= R::value ? arg[Is - R::value] : false)...));
            }(arg, iota_sequence<LE, LA>);
        }
    }

    template <typename L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> arg, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_mask<L, A>;
        return []<size_t... Is>(
                   mask_type arg, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>(
                bitset((Is >= shift ? arg[Is - shift] : false)...));
        }(arg, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_bwsl<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwsl<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_bwsl<basic_mask<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwsl<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <extended_vector T>
    requires unqualified_bwsl<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_bwsl<T>) {
            return bwshift_left(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <extended_mask T>
    requires unqualified_masksl<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_bwsl<T>) {
            return bwshift_left(val, shift);
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
        if constexpr (unqualified_canonical_bwsli<basic_vector<E, A>, Shift>) {
            if consteval {
                return operator()(val, Shift::value);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return operator()(val, Shift::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    requires unqualified_canonical_bwsli<basic_vector<E, A>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, Shift shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_bwsli<basic_mask<E, A>, Shift>) {
            if consteval {
                return operator()(val, Shift::value);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return operator()(val, Shift::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E,
        integral_constant_like Shift>
    requires unqualified_canonical_bwsli<basic_mask<E, A>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, Shift shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <extended_vector T, integral_constant_like Shift>
    requires unqualified_bwsli<T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Shift shift) noexcept {
        if constexpr (unqualified_extended_bwsli<T, Shift>) {
            return bwshift_left(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <extended_mask T, integral_constant_like Shift>
    requires unqualified_masksli<T, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Shift shift) noexcept {
        if constexpr (unqualified_extended_bwsli<T, Shift>) {
            return bwshift_left(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    //

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE> && integral<LE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(
        basic_vector<LE, A> val, basic_vector<RE, A> shift) noexcept {
        if constexpr (unqualified_canonical_bwslv<basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <simd_abi LA, simd_element_for<LA> LE, common_abi_with<LA> RA,
        simd_element_for<RA> RE>
    requires unqualified_canonical_bwslv<basic_vector<LE, LA>,
        basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, common_abi_t<LA, RA>> operator()(
        basic_vector<LE, LA> val, basic_vector<RE, RA> shift) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        common_abi_with<typename L::abi_type, typename R::abi_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        integral<typename R::value_type> && unqualified_bwslv<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, R shift) noexcept {
        if constexpr (unqualified_extended_bwslv<L, R>) {
            return bwshift_left(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    ///
    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<MaskE, A> mask, basic_vector<E, A> val,
        size_t shift) noexcept {
        if constexpr (unqualified_canonical_mbwsl<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(pass, mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, pass, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2>) &&
        simd_element_for<E, A2> &&
        maskable_args<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>> &&
        unqualified_canonical_mbwsl<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A1> operator()(basic_vector<E, A1> pass,
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val,
        size_t shift) noexcept {
        return bwshift_left(internal::abi<A1>, pass, mask, val, shift);
    }

    template <simd_vector Pass, simd_mask Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<Pass, Mask, Arg> &&
        extended_mbwsl<bwshift_left_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        Pass pass, Mask mask, Arg arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<Pass, Mask, Arg>) {
            return bwshift_left(pass, mask, arg, shift);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_mbwsl<zero_t, basic_mask<MaskE, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                return bwshift_left(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2>) &&
        simd_element_for<E, A2> &&
        zmaskable_args<basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mbwsl<dx::zero_t, basic_mask<MaskE, A1>,
            basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, A1> mask,
        basic_vector<E, A2> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<A1>, dx::zero, mask, val, shift);
    }

    template <simd_mask Mask, simd_vector Arg>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_mbwsl<bwshift_left_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        Mask mask, Arg arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<dx::zero_t, Mask, Arg>) {
            return bwshift_left(mask, arg, shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(arg), shift);
        }
    }

    template <simd_mask Mask, simd_vector Arg>
    requires requires(Mask mask, Arg arg, size_t shift) {
        bwshift_left_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, Arg arg, size_t shift) noexcept {
        return operator()(mask, arg, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        Mask mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_imbwsl<basic_vector<E, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(pass, mask, val, shift);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return bwshift_left<V>(
                    internal::abi<A>, pass, masked_operation, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi PassA, simd_element_for<PassA> E,
        const_mask_for<basic_vector<E, PassA>> Mask, simd_abi InA>
    requires (different_from<PassA, InA> || scalable_abi<PassA> ||
                 scalable_abi<InA>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, PassA>, basic_vector<E, InA>> &&
        unqualified_canonical_imbwsl<basic_vector<E, PassA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, PassA> operator()(
        basic_vector<E, PassA> pass, Mask mask, basic_vector<E, InA> val,
        size_t shift) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, PassA>, Mask>;
        return bwshift_left<V>(
            internal::abi<PassA>, pass, masked_operation, val, shift);
    }

    template <simd_vector Pass, const_mask_for<Pass> Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_vector<Arg>) &&
        imm_maskable_args<Pass, Arg> &&
        extended_imbwsl<bwshift_left_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        Pass pass, Mask mask, Arg arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<Pass, Mask, Arg>) {
            constexpr auto V = const_mask_v<Pass, Mask>;
            return bwshift_left<V>(pass, masked_operation, arg, shift);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<Pass>(mask), dx::to_canonical(arg),
                shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_imbwsl<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return bwshift_left<V>(
                    internal::abi<A>, dx::zero, masked_operation, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask>
    requires scalable_abi<InA> && simd_element_for<E, InA> &&
        imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imbwsl<zero_t, Mask, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        Mask mask, basic_vector<E, InA> val, size_t shift) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, InA>, Mask>;
        return bwshift_left<V>(
            internal::abi<InA>, dx::zero, masked_operation, val, shift);
    }

    template <extended_vector Arg, const_mask_for<Arg> Mask>
    requires imm_zmaskable_args<Arg> &&
        extended_imbwsl<bwshift_left_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        Mask mask, Arg arg, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<zero_t, Mask, Arg>) {
            constexpr auto V = const_mask_v<Arg, Mask>;
            return bwshift_left<V>(dx::zero, masked_operation, arg, shift);
        } else {
            return operator()(dx::to_compatible_const_mask<Arg>(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <simd_vector Arg, const_mask_for<Arg> Mask>
    requires requires(Mask mask, Arg arg, size_t shift) {
        bwshift_left_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, Arg arg, size_t shift) noexcept {
        return operator()(mask, arg, shift);
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE, integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<MaskE, A> mask, basic_vector<E, A> val,
        Shift shift) noexcept {
        if constexpr (unqualified_canonical_mbwsli<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_left_t>(pass, mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, pass, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2, integral_constant_like Shift>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2>) &&
        simd_element_for<E, A2> &&
        maskable_args<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>> &&
        unqualified_canonical_mbwsli<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A1> operator()(basic_vector<E, A1> pass,
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val,
        Shift shift) noexcept {
        return bwshift_left(internal::abi<A1>, pass, mask, val, shift);
    }

    template <simd_vector Pass, simd_mask Mask, simd_vector Arg,
        integral_constant_like Shift>
    requires (extended_vector<Pass> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<Pass, Mask, Arg> &&
        extended_mbwsli<bwshift_left_t, Pass, Mask, Arg, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        Pass pass, Mask mask, Arg arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<Pass, Mask, Arg, Shift>) {
            return bwshift_left(pass, mask, arg, shift);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE, integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_mbwsli<zero_t, basic_mask<MaskE, A>,
                          basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                return bwshift_left(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2, integral_constant_like Shift>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2>) &&
        simd_element_for<E, A2> &&
        zmaskable_args<basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mbwsli<dx::zero_t, basic_mask<MaskE, A1>,
            basic_vector<E, A2>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, A1> mask,
        basic_vector<E, A2> val, Shift shift) noexcept {
        return bwshift_left(internal::abi<A1>, dx::zero, mask, val, shift);
    }

    template <simd_mask Mask, simd_vector Arg, integral_constant_like Shift>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_mbwsli<bwshift_left_t, zero_t, Mask, Arg, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<dx::zero_t, Mask, Arg,
                          Shift>) {
            return bwshift_left(mask, arg, shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(arg), shift);
        }
    }

    template <simd_mask Mask, simd_vector Arg, integral_constant_like Shift>
    requires requires(Mask mask, Arg arg, Shift shift) {
        bwshift_left_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, Arg arg, Shift shift) noexcept {
        return operator()(mask, arg, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask, integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        Mask mask, basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_imbwsli<basic_vector<E, A>, Mask,
                          basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_left_t>(pass, mask, val, shift);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return bwshift_left<V>(
                    internal::abi<A>, pass, masked_operation, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(pass, mask, val, shift);
        }
    }

    template <simd_abi PassA, simd_element_for<PassA> E,
        const_mask_for<basic_vector<E, PassA>> Mask, simd_abi InA,
        integral_constant_like Shift>
    requires (different_from<PassA, InA> || scalable_abi<PassA> ||
                 scalable_abi<InA>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, PassA>, basic_vector<E, InA>> &&
        unqualified_canonical_imbwsli<basic_vector<E, PassA>, Mask,
            basic_vector<E, InA>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, PassA> operator()(
        basic_vector<E, PassA> pass, Mask mask, basic_vector<E, InA> val,
        Shift shift) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, PassA>, Mask>;
        return bwshift_left<V>(
            internal::abi<PassA>, pass, masked_operation, val, shift);
    }

    template <simd_vector Pass, const_mask_for<Pass> Mask, simd_vector Arg,
        integral_constant_like Shift>
    requires (extended_vector<Pass> || extended_vector<Arg>) &&
        imm_maskable_args<Pass, Arg> &&
        extended_imbwsli<bwshift_left_t, Pass, Mask, Arg, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        Pass pass, Mask mask, Arg arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<Pass, Mask, Arg, Shift>) {
            constexpr auto V = const_mask_v<Pass, Mask>;
            return bwshift_left<V>(pass, masked_operation, arg, shift);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<Pass>(mask), dx::to_canonical(arg),
                shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask, integral_constant_like Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val, Shift shift) noexcept {
        if constexpr (unqualified_canonical_imbwsli<zero_t, Mask,
                          basic_vector<E, A>, Shift>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return bwshift_left<V>(
                    internal::abi<A>, dx::zero, masked_operation, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask, integral_constant_like Shift>
    requires scalable_abi<InA> && simd_element_for<E, InA> &&
        imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imbwsli<zero_t, Mask, basic_vector<E, InA>, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        Mask mask, basic_vector<E, InA> val, Shift shift) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, InA>, Mask>;
        return bwshift_left<V>(
            internal::abi<InA>, dx::zero, masked_operation, val, shift);
    }

    template <extended_vector Arg, const_mask_for<Arg> Mask,
        integral_constant_like Shift>
    requires imm_zmaskable_args<Arg> &&
        extended_imbwsli<bwshift_left_t, zero_t, Mask, Arg, Shift>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg, Shift shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<zero_t, Mask, Arg, Shift>) {
            constexpr auto V = const_mask_v<Arg, Mask>;
            return bwshift_left<V>(dx::zero, masked_operation, arg, shift);
        } else {
            return operator()(dx::to_compatible_const_mask<Arg>(mask),
                dx::to_canonical(arg), shift);
        }
    }

    template <simd_vector Arg, const_mask_for<Arg> Mask,
        integral_constant_like Shift>
    requires requires(Mask mask, Arg arg, Shift shift) {
        bwshift_left_t::operator()(mask, arg, shift);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, Arg arg, Shift shift) noexcept {
        return operator()(mask, arg, shift);
    }
    ///
    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> MaskE>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_vector<LE, A> src,
        basic_mask<MaskE, A> mask, basic_vector<LE, A> lhs,
        basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwslv<bwshift_left_t,
                          basic_vector<LE, A>, basic_mask<MaskE, A>,
                          basic_vector<LE, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_left(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> LE, common_size_with<LE> MaskE,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> && common_size_with<LE, RE> &&
        (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
            scalable_abi<RA>) &&
        simd_element_for<LE, LA> &&
        maskable_args<basic_vector<LE, SA>, basic_mask<MaskE, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_mbwslv<bwshift_left_t, basic_vector<LE, SA>,
            basic_mask<MaskE, SA>, basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<MaskE, SA> mask, basic_vector<LE, LA> lhs,
        basic_vector<RE, RA> rhs) noexcept {
        return bwshift_left(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask Mask, simd_vector L, simd_vector R>
    requires integral<typename R::value_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        (extended_vector<S> || extended_mask<Mask> || extended_vector<L> ||
            extended_vector<R>) &&
        maskable_args<S, Mask, L, R> &&
        extended_mbwslv<bwshift_left_t, S, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwslv<bwshift_left_t, S, Mask, L,
                          R>) {
            return bwshift_left(src, mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> MaskE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<LE, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwslv<bwshift_left_t, zero_t,
                          basic_mask<MaskE, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, lhs, rhs);
            } else {
                return bwshift_left(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::broadcast<LE, A>(dx::zero), mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<MaskE, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mbwslv<bwshift_left_t, zero_t,
            basic_mask<MaskE, SA>, basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_left(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask Mask, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<Mask> || extended_vector<L> ||
                 extended_vector<R>) &&
        zmaskable_args<Mask, L, R> &&
        extended_mbwslv<bwshift_left_t, zero_t, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwslv<bwshift_left_t, zero_t, Mask,
                          L, R>) {
            return bwshift_left(mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask Mask, simd_vector L, simd_vector R>
    requires requires(
        Mask mask, L lhs, R rhs) { bwshift_left_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imbwslv<bwshift_left_t,
                          basic_vector<E, A>, Mask, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return bwshift_left<V>(
                    internal::abi<A>, src, masked_operation, lhs, rhs);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi LA,
        common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_imbwslv<bwshift_left_t, basic_vector<E, SA>, Mask,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, SA>, Mask>;
        return bwshift_left<V>(
            internal::abi<SA>, src, masked_operation, lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector L,
        simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> &&
        extended_imbwslv<bwshift_left_t, S, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imbwslv<bwshift_left_t, S, Mask, L,
                          R>) {
            constexpr auto V = const_mask_v<S, Mask>;
            return bwshift_left<V>(src, masked_operation, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imbwslv<bwshift_left_t, zero_t,
                          Mask, basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, lhs, rhs);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return bwshift_left<V>(
                    internal::abi<A>, dx::zero, masked_operation, lhs, rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> Mask>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imbwslv<bwshift_left_t, zero_t, Mask,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
        return bwshift_left<V>(
            internal::abi<A>, dx::zero, masked_operation, lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwshift_left_t, L, R>> Mask>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> &&
        extended_imbwslv<bwshift_left_t, zero_t, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<bwshift_left_t, L, R>;
        if constexpr (unqualified_extended_imbwslv<bwshift_left_t, zero_t, Mask,
                          L, R>) {
            constexpr auto V = const_mask_v<S, Mask>;
            return bwshift_left<V>(dx::zero, masked_operation, lhs, rhs);
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R, const_mask_like Mask>
    requires requires(
        Mask mask, L lhs, R rhs) { bwshift_left_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

template <size_t N>
struct bwshift_lefti_t {
    template <typename... Args>
    requires invocable<bwshift_left_t, Args..., size_constant<N>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args... val) noexcept {
        constexpr size_constant<N> shift;
        return bwshift_left_t::operator()(val..., shift);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwshift_left_t bwshift_left{};
DPL_EXPORT template <size_t N>
inline constexpr internal::bwshift_lefti_t<N> bwshift_lefti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
