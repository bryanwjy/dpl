// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/packed_indices.h"
#include "dpl/core/algorithm/permute.h"

#if !DPL_MODULES
#  include "dpl/core/basic/const_mask.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/to_native_type.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/logic.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename F, typename T>
concept reduction_operator_for = simd_vector<T> && regular_invocable<F, T, T> &&
    core_convertible_to<invoke_result_t<F, T, T>, T>;

template <typename T, auto V>
concept reducible = simd_vector<T> && integral<decltype(V)> &&
    (V == static_cast<decltype(V)>(-1) ||
        __DPL bit_width(__DPL to_unsigned(V)) <= simd_abi_traits<T>::size);

template <typename T, typename BinaryOp>
concept unqualified_reduce = requires(T val, BinaryOp op) {
    { reduce(internal::abi<T>, val, op) } -> equivalent_simd_as<T>;
};

template <typename M, typename T, typename BinaryOp>
concept unqualified_mreduce = requires(M mask, T val, BinaryOp op) {
    { reduce(internal::abi<T>, mask, val, op) } -> equivalent_simd_as<T>;
};

template <typename M, typename T, typename BinaryOp>
concept unqualified_reducei =
    const_mask_for<M, T> && requires(T val, BinaryOp op) {
        {
            reduce<const_mask_v<T, M>>(internal::abi<T>, val, op)
        } -> equivalent_simd_as<T>;
    };

struct reduce_t {
private:
    template <const_mask_like M>
    static constexpr size_t accumulations = __DPL popcount(dx::popcount(M{}));
    template <simd_vector T, reduction_operator_for<T> BinaryOp>
    static constexpr bool is_nothrow_v = is_nothrow_invocable_v<BinaryOp, T, T>;

    template <typename M, simd_vector T, typename BinaryOp>
    requires (accumulations<M> == 1)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto fallbacki(M mask, T value, BinaryOp&& op) noexcept(
        is_nothrow_v<T, BinaryOp>) {
        auto const reducer = [&]<size_t I = 0>(this auto const self, T value,
                                 immediate<I> = {}) {
            constexpr seq::packed_indices<simd_abi_traits<T>::size> iota{};
            if constexpr (I == dx::countr_zero(dx::popcount(mask))) {
                return value;
            } else {
                constexpr auto idx = seq::rotate(mask, iota, 1zu << I);
                auto rhs = [&]<size_t... Is>(__DPL index_sequence<Is...>) {
                    return dx::permutei<idx[Is]...>(value);
                }(iota_sequence<T>);
                return self( __DPL invoke_r<T>(op, value, rhs), imm<I + 1>);
            }
        };

        return reducer(value);
    }

    template <typename M, simd_vector T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto fallbacki(M mask, T value, BinaryOp&& op) noexcept(
        is_nothrow_v<T, BinaryOp>) {
        static_assert(accumulations<M> > 1);
        // (Comment re-written by Gemini for better clarity)
        // Logic: Decompose 'active' elements into power-of-two chunks based on
        // its binary form.
        // Example: 11 elements (0b1011)
        // 1. Primary Tree: Reduces the largest power-of-two (2^3 = 8).
        // 2. Remainder: Accumulates partial reductions from levels where bits
        // are set (2^1 and 2^0).
        //
        // Complexity:
        // - Tree Depth: floor(log2(active)) shuffles.
        // - Merging: (popcnt(active) - 1) additional operations.
        // - Final: 1 permutation to align the remainder with the primary tree
        // result.

        static constexpr auto active = dx::popcount(mask);
        static constexpr auto activity_width = sizeof(active) * char_bit_v;
        struct nothing_t {};
        union remainder_t {
            nothing_t null;
            T value;
        };

        auto const reducer = [&]<size_t I = 0>(this auto const self, T lhs,
                                 remainder_t remain = {.null = {}},
                                 immediate<I> = {}) -> T {
            constexpr seq::packed_indices<simd_abi_traits<T>::size> iota{};
            if constexpr (I + 1 == (activity_width - dx::countl_zero(active))) {
                constexpr auto idx = seq::rotate(mask, iota, (1zu << I) - 1);
                remain.value = [&]<size_t... Is>( __DPL index_sequence<Is...>) {
                    return dx::permutei<idx[Is]...>(remain.value);
                }(iota_sequence<T>);
                return __DPL invoke_r<T>(op, value, remain.value);
            } else {
                constexpr auto idx = seq::rotate(mask, iota, 1zu << I);
                auto const rhs = [&]<size_t... Is>(
                                     __DPL index_sequence<Is...>) {
                    return dx::permutei<idx[Is]...>(value);
                }(iota_sequence<T>);
                if constexpr (active & (1zu << I)) {
                    if constexpr (I == __DPL countr_zero(active)) {
                        remain = remainder_t{.value = rhs};
                    } else {
                        remain.value = __DPL invoke_r<T>(op, remain.value, rhs);
                    }
                }
                return self(
                    __DPL invoke_r<T>(op, lhs, rhs), remain, imm<I + 1>);
            }
        };

        return reducer(value);
    }

public:
    template <fixed_width_vector T, reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T value, BinaryOp op) noexcept {
        constexpr auto all = make_const_mask_t<T, -1>{};
        if constexpr (unqualified_reduce<T, BinaryOp>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(reduce(
                        internal::abi<T>, value, op))::value_type;
                    return dx::reinterpret<E>(fallbacki(all, value, op));
                } else {
                    return reduce(internal::abi<T>, value, op);
                }
            } else {
                return reduce(internal::abi<T>, value, op);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallbacki(all, value, op);
        } else {
            return operator()(dx::to_canonical(value), op);
        }
    }

    template <fixed_width_vector T, const_mask_for<T> M,
        reduction_operator_for<T> BinaryOp>
    requires reducible<T, M::value>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T value, BinaryOp op) noexcept(
        is_nothrow_v<T, BinaryOp>) {
        if constexpr (unqualified_reduce<T, BinaryOp>) {
            constexpr auto V = const_mask_v<T, M>;
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(reduce<V>(
                        internal::abi<T>, value, op))::value_type;
                    return dx::reinterpret<E>(fallbacki(mask, value, op));
                } else {
                    return reduce<V>(internal::abi<T>, value, op);
                }
            } else {
                return reduce<V>(internal::abi<T>, value, op);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallbacki(mask, value, op);
        } else {
            return operator()(mask, dx::to_canonical(value), op);
        }
    }

    template <fixed_width_vector T, compatible_mask_with<T> M,
        reduction_operator_for<T> BinaryOp>
    requires reducible<T, M::value> &&
        (unqualified_mreduce<M, T, BinaryOp> ||
            unqualified_mreduce<canonical_type_t<M>, canonical_type_t<T>,
                BinaryOp>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T value, BinaryOp op) noexcept(
        is_nothrow_v<T, BinaryOp>) {
        // TODO: Investigate if a performant fallback is even possible
        if constexpr (unqualified_mreduce<M, T, BinaryOp>) {
            return reduce(internal::abi<T>, mask, value, op);
        } else {
            return reduce(internal::abi<T>, dx::to_canonical(mask),
                dx::to_canonical(value), op);
        }
    }
};

template <auto V>
struct reducei_t {};

template <integral auto V>
struct reducei_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_const_mask_t<T, V>;

public:
    template <arithmetic_vector T>
    requires fixed_width_vector<T> && requires {
        typename mask_type<T>;
        requires regular_invocable<reduce_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return reduce_t::operator()(mask, arg);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <auto V>
inline constexpr internal::reducei_t<V> reducei{};
DPL_EXPORT inline constexpr internal::reduce_t reduce{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
