// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/packed_indices.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/to_native_type.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/logic.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT struct predication_t {
    __DPL_HIDE_FROM_ABI explicit constexpr predication_t() noexcept = default;
};
DPL_EXPORT struct adaptive_t {
    __DPL_HIDE_FROM_ABI explicit constexpr adaptive_t() noexcept = default;
};

DPL_EXPORT inline constexpr predication_t predication{};
DPL_EXPORT inline constexpr adaptive_t adaptive{};

} // namespace datapar

namespace datapar::internal {

template <simd_type T, auto V>
class reduction_result {
    using vector_type =
        typename T::abi_type::template native_type<typename T::value_type>;

public:
    using value_type = typename T::value_type;
    using abi_type = typename T::abi_type;

    __DPL_HIDE_FROM_ABI constexpr reduction_result() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr reduction_result(T result) noexcept
        : result_(result) {}
    __DPL_HIDE_FROM_ABI constexpr reduction_result(vector_type vec) noexcept
        : result_(vec) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    explicit constexpr DPL_VECTORCALL operator vector_type(
        this reduction_result self) noexcept {
        return datapar::to_native_type(self.result_);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    explicit constexpr DPL_VECTORCALL operator T(
        this reduction_result self) noexcept {
        return self.result_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr auto operator[](
        this reduction_result self, extraction_index auto idx) noexcept {
        constexpr immediate_mask<element_count<T>, V> mask{};
        if constexpr (integral_constant_like<decltype(idx)>) {
            static_assert(mask[idx]);
        }
        return self.result_[idx];
    }

    template <integral auto O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr auto DPL_VECTORCALL
        bit_keep(abi_type, reduction_result self) noexcept {
        static constexpr immediate_mask<element_count<T>, V> mask{};
        static constexpr auto not_zpos = dx::countr_zero(mask);
        if constexpr (V == O) {
            return self.result_;
        } else if ((V | O) == V) {
            return dx::bit_keepi<O>(self.result_);
        } else if constexpr (dx::all_of(mask)) {
            return dx::broadcast_lanei<not_zpos>(self.result_);
        } else {
            return dx::bit_keepi<O>(
                dx::broadcast_lanei<not_zpos>(self.result_));
        }
    }

    template <integral auto O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr auto DPL_VECTORCALL
        bit_drop(abi_type abi, reduction_result self) noexcept {
        static constexpr immediate_mask<element_count<T>, V> mask{};
        static constexpr auto inv = decltype(~mask)::value;
        return bit_keep<inv>(abi, self);
    }

private:
    T result_;
};

template <typename F, typename T>
concept reduction_operator_for = simd_type<T> && regular_invocable<F, T, T> &&
    core_convertible_to<invoke_result_t<F, T, T>, T>;

template <typename T, auto V>
concept reducible = simd_type<T> && integral<decltype(V)> &&
    (V == static_cast<decltype(V)>(-1) ||
        __DPL bit_width(__DPL to_unsigned(V)) <= element_count<T>);

struct reduce_t {
private:
    template <immediate_mask_like M>
    static constexpr size_t accumulations = __DPL popcount(dx::popcount(M{}));
    template <simd_type T, reduction_operator_for<T> BinaryOp>
    static constexpr bool is_nothrow_v = is_nothrow_invocable_v<BinaryOp, T, T>;

public:
    template <simd_type T, reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T val, BinaryOp op) noexcept {
        constexpr auto all = immediate_mask<element_count<T>, -1>{};
        return operator()(all, val, op);
    }

    /**
     * Reduction for power of two elements
     */
    template <simd_type T, immediate_mask_for<T> M,
        reduction_operator_for<T> BinaryOp>
    requires reducible<T, M::value> && (accumulations<M> == 1)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T value, BinaryOp op) noexcept(
        is_nothrow_v<T, BinaryOp>) {
        auto const reducer =
            [&]<size_t I = 0>(this auto const self, T value,
                immediate<I> = {}) noexcept(is_nothrow_v<T, BinaryOp>) {
                constexpr seq::packed_indices<element_count<T>> iota{};
                if constexpr (I == dx::countr_zero(dx::popcount(mask))) {
                    return value;
                } else {
                    constexpr auto idx = seq::rotate(mask, iota, 1zu << I);
                    auto rhs = [&]<size_t... Is>(__DPL index_sequence<Is...>) {
                        return dx::permutei<idx[Is]...>(value);
                    }(iota_sequence<T>);
                    return self( __DPL invoke(op, value, rhs), imm<I + 1>);
                }
            };

        return reduction_result<T, M::value>(reducer(value));
    }

    /**
     * Reduction for non-power of two elements
     */
    template <simd_type T, immediate_mask_for<T> M,
        reduction_operator_for<T> BinaryOp>
    requires reducible<T, M::value>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T value, BinaryOp op) noexcept(
        is_nothrow_v<T, BinaryOp>) {
        static_assert(accumulations<M> > 1);
        static constexpr auto active = dx::popcount(mask);
        static constexpr auto activity_width = sizeof(active) * char_bit_v;
        // The way the logic works is dependent on the binary number
        // representation of 'active'. e.g. For 11 elements, the binary
        // representation is 0b1011.

        // An additional buffer is requires to accumulate a 'remainder' The
        // remainder consists of the accumulation of the permuted vectors of
        // each level (except the last) in the shuffle tree where the
        // corresponding binary representation of the active element count is
        // set. The final result is then:
        //  op(reduction[depth], rotate(mask, remainder, (1 << depth) - 1))
        // where depth == std::bit_width(active) - 1
        // and reduction[depth] is the reduction result using the shuffle tree
        // up to and including the depth of std::bit_width(active) - 1

        // So the maximum 'complexity' of non-power of two elements is
        // essentially:
        // Reduction of `depth` elements + popcnt(active % depth) op + 1 permute
        // e.g. 11 elements: reduction of 8 elements + 2 op and 1 permute

        struct nothing_t {};
        union remainder_t {
            nothing_t null;
            T value;
        };

        auto const reducer =
            [&]<size_t I = 0>(this auto const self, T lhs,
                remainder_t remain = {.null = {}},
                immediate<I> = {}) noexcept(is_nothrow_v<T, BinaryOp>) -> T {
            constexpr seq::packed_indices<element_count<T>> iota{};
            if constexpr (I + 1 == (activity_width - dx::countl_zero(active))) {
                constexpr auto idx = seq::rotate(mask, iota, (1zu << I) - 1);
                remain.value = [&]<size_t... Is>( __DPL index_sequence<Is...>) {
                    return dx::permutei<idx[Is]...>(remain.value);
                }(iota_sequence<T>);
                return __DPL invoke(op, value, remain.value);
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
                        remain.value = __DPL invoke(op, remain.value, rhs);
                    }
                }
                return self( __DPL invoke(op, lhs, rhs), remain, imm<I + 1>);
            }
        };

        return reduction_result<T, M::value>(reducer(value));
    }
};

template <auto V>
struct reducei_t {};

template <integral auto V>
struct reducei_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <arithmetic_simd T>
    requires fixed_width_simd<T> && requires {
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
DPL_EXPORT template <simd_type T, integral auto V>
inline constexpr bool enable_simd_type<internal::reduction_result<T, V>> = true;
inline namespace cpo {
DPL_EXPORT template <integral auto V>
inline constexpr internal::reducei_t<V> reducei{};
DPL_EXPORT inline constexpr internal::reduce_t reduce{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
