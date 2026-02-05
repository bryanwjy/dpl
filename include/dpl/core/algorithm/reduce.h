// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/logic.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/utility/packed_indices.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace reduction {
template <size_t N>
struct shift_t;
template <size_t I>
inline constexpr shift_t<I> shift_v{};
struct shift_result {
    int first, second;
};
} // namespace reduction

template <simd_type T, auto V>
class reduction_result {
    using vector_type =
        typename T::abi_type::template native_type<typename T::value_type>;

public:
    using value_type = typename T::value_type;
    using abi_type = typename T::abi_type;

    __DPL_HIDE_FROM_ABI constexpr reduction_result() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr reduction_result(vector_type vec) noexcept
        : result_(vec) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(
        this reduction_result self) noexcept {
        return +self.result_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr DPL_VECTORCALL operator T(
        this reduction_result self) noexcept {
        return self.result_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto operator[](
        this reduction_result self, extraction_index auto idx) noexcept {
        return self.result_[idx];
    }

    template <integral auto O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr auto DPL_VECTORCALL
        bit_keep(abi_type, reduction_result self) noexcept {
        static constexpr immediate_mask<element_count<T>, V> mask{};
        static constexpr auto not_zpos = __DPL countr_zero(V);
        if constexpr (V == O) {
            return self.result_;
        } else if ((V | O) == V) {
            return dx::bit_keepi<O>(self.result_);
        } else if constexpr (dx::all_of(mask)) {
            return dx::permute<not_zpos, not_zpos, not_zpos, not_zpos>(
                self.result_);
        } else {
            return dx::bit_keepi<O>(
                dx::permute<not_zpos, not_zpos, not_zpos, not_zpos>(
                    self.result_));
        }
    }

    template <integral auto O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr auto DPL_VECTORCALL
        bit_drop(abi_type abi, reduction_result self) noexcept {
        static constexpr immediate_mask<element_count<T>, V> mask{};
        static constexpr auto not_zpos = __DPL countr_zero(V);
        return bit_keep<decltype(~mask)::value>(abi, self);
    }

private:
    T result_;
};

template <typename F, typename T>
concept reduction_operator_for = simd_type<T> && regular_invocable<F, T, T> &&
    (same_as<T, invoke_result_t<F, T, T>> ||
        requires(invoke_result_t<F, T, T> arg) {
            requires equivalent_simd_as<T, invoke_result_t<F, T, T>>;
            dx::reinterpret<T>(arg);
        });

template <typename T, auto V>
concept reducible = simd_type<T> && integral<decltype(V)> &&
    (V == static_cast<decltype(V)>(-1) ||
        __DPL bit_width(__DPL to_unsigned(V)) <= element_count<T>);

template <auto V>
struct reducei_t {};

template <integral auto V>
struct reducei_t<V> {
    template <simd_type T, reduction_operator_for<T> BinaryOp>
    static constexpr bool is_nothrow_v = is_nothrow_invocable_v<BinaryOp, T, T>;

    template <reducible<V> T, reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T value, BinaryOp) noexcept(
        is_nothrow_v<T, BinaryOp>)
    requires (__DPL popcount(__DPL to_unsigned(V)) == 0)
    {
        return value;
    }

    template <reducible<V> T, reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T value, BinaryOp) noexcept
    requires (__DPL popcount(__DPL to_unsigned(V)) == 1)
    {
        constexpr auto idx =
            static_cast<size_t>(__DPL countr_zero(__DPL to_unsigned(V)));
        return dx::broadcast_element<idx>(value);
    }

    template <reducible<V> T, reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T value, BinaryOp op) noexcept(is_nothrow_v<T, BinaryOp>) {
        using indices_type = packed_indices<element_count<T>>;
        static constexpr immediate_mask<element_count<T>, V> imm_mask;
        static constexpr auto popcount = imm<dx::popcount(imm_mask)>;
        // TODO: > 16
        static_assert(popcount <= 16, "Currently unsupported");
        auto const reducer =
            [&]<int I>(this auto const self, immediate<I>) noexcept(
                is_nothrow_v<T, BinaryOp>) {
                if constexpr (I == 1) {
                    return value;
                } else {
                    constexpr auto shifts =
                        reduction::shift_v<popcount>(imm<I>);
                    constexpr auto idx = shifts.first < 0
                        ? indices_type().template rotate_left<V>(shifts.first)
                        : indices_type().template rotate_right<V>(shifts.first);

                    auto const left = self(imm<shifts.first>);
                    auto const right = self(imm<shifts.second>);

                    return [&]<size_t... Is>(__DPL index_sequence<Is...>) {
                        return dx::reinterpret<T>(__DPL invoke(
                            op, left, dx::permute<idx[Is]...>(right)));
                    }(iota_sequence<T>);
                }
            };

        using R = reduction_result<T, V>;
        return R(+reducer(popcount));
    }
};

struct reduce_t {
    template <simd_type T, immediate_mask_for<T> M,
        reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, BinaryOp op) noexcept
    requires reducible<T, decltype(dx::to_immediate_mask<T>(mask))::value>
    {
        constexpr auto V = dx::immediate_mask_v<T, M>;
        return reducei_t<V>::operator()(val, op);
    }

    template <simd_type T, reduction_operator_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, BinaryOp op) noexcept {
        return reducei_t<-1>::operator()(val, op);
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

namespace internal::reduction {

#define __DPL_REDUCTION_SHIFT(RESULT, LEFT, RIGHT)                         \
    static consteval shift_result operator()(immediate<RESULT>) noexcept { \
        return {LEFT, RIGHT};                                              \
    }                                                                      \
    static_assert(RESULT == LEFT + RIGHT);                                 \
    static_assert(LEFT >= RIGHT)

template <>
struct shift_t<2> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
};

template <>
struct shift_t<3> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
};

template <>
struct shift_t<4> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(4, 2, 2);
};

template <>
struct shift_t<5> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(5, 3, 2);
};
template <>
struct shift_t<6> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(6, 3, 3);
};
template <>
struct shift_t<7> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(4, 2, 2);
    __DPL_REDUCTION_SHIFT(7, 4, 3);
};
template <>
struct shift_t<8> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(4, 2, 2);
    __DPL_REDUCTION_SHIFT(8, 4, 4);
};
template <>
struct shift_t<9> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(6, 3, 3);
    __DPL_REDUCTION_SHIFT(9, 6, 3);
};
template <>
struct shift_t<10> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(5, 3, 2);
    __DPL_REDUCTION_SHIFT(10, 5, 5);
};
template <>
struct shift_t<11> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(4, 2, 2);
    __DPL_REDUCTION_SHIFT(7, 4, 3);
    __DPL_REDUCTION_SHIFT(11, 7, 4);
};
template <>
struct shift_t<12> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(6, 3, 3);
    __DPL_REDUCTION_SHIFT(12, 6, 6);
};
template <>
struct shift_t<13> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(5, 3, 2);
    __DPL_REDUCTION_SHIFT(8, 5, 3);
    __DPL_REDUCTION_SHIFT(13, 8, 5);
};
template <>
struct shift_t<14> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(4, 2, 2);
    __DPL_REDUCTION_SHIFT(7, 4, 3);
    __DPL_REDUCTION_SHIFT(14, 7, 7);
};
template <>
struct shift_t<15> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(3, 2, 1);
    __DPL_REDUCTION_SHIFT(5, 3, 2);
    __DPL_REDUCTION_SHIFT(10, 5, 5);
    __DPL_REDUCTION_SHIFT(15, 10, 5);
};
template <>
struct shift_t<16> {
    __DPL_REDUCTION_SHIFT(2, 1, 1);
    __DPL_REDUCTION_SHIFT(4, 2, 2);
    __DPL_REDUCTION_SHIFT(8, 4, 4);
    __DPL_REDUCTION_SHIFT(16, 8, 8);
};

#undef __DPL_REDUCTION_SHIFT
} // namespace internal::reduction
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
