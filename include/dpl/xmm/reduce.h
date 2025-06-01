// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/immediate_mask.h"
#include "dpl/xmm/bitset.h"
#include "dpl/xmm/blob.h"
#include "dpl/xmm/concepts.h"
#include "dpl/xmm/max.h"
#include "dpl/xmm/min.h"
#include "dpl/xmm/vector.h"

#if !DPL_MODULES
#  include <immintrin.h>

#  include <bit>
#  include <concepts>
#  include <functional>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

namespace details::reduce {

template <size_t N>
struct shift_t;

template <size_t I>
inline constexpr shift_t<I> shift_v{};

struct shift_result {
    int first, second;
};

DPL_EXPORT template <xmm_element T, bit_type_for<T> V>
class reduction_result : public unary_vector_expression {

public:
    using element_type = T;
    using native_type = native_type_t<T>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline reduction_result(
        vector<T> result) noexcept
        : result_{result} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    evaluate(reduction_result self) noexcept {
        return self.result_;
    }

    template <std::integral auto O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    select(reduction_result self, mask_for<T, O> = {}) noexcept {
        static constexpr auto not_zpos = std::countr_zero(V);
        if constexpr (V == O) {
            return self.result_;
        } else if ((V | O) == V) {
            return select<O>(self.result_);
        } else if constexpr (O == +bitset_for<T>{all<T>}) {
            return permute<not_zpos, not_zpos, not_zpos, not_zpos>(
                self.result_);
        } else {
            return select<O>(
                permute<not_zpos, not_zpos, not_zpos, not_zpos>(self.result_));
        }
    }

private:
    vector<T> result_;
};

template <typename T, typename E>
concept vector_binary_invocable_for =
    xmm_element<E> && std::regular_invocable<T, vector<E>, vector<E>> &&
    std::convertible_to<std::invoke_result_t<T, vector<E>, vector<E>>,
        vector<E>>;

template <xmm_element T, bit_type_for<T> V>
struct reduce_t {
    static constexpr auto popcount = immediate_v<std::popcount(V)>;
    template <vector_binary_invocable_for<T> BinaryOp>
    static constexpr bool is_nothrow_v =
        std::is_nothrow_invocable_v<BinaryOp, vector<T>, vector<T>>;

    template <vector_binary_invocable_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) static inline auto
    operator()(vector<T> value, BinaryOp op) noexcept(is_nothrow_v<BinaryOp>)
    requires (popcount == 0)
    {
        return value;
    }

    template <vector_binary_invocable_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) static inline auto
    operator()(vector<T> value, BinaryOp op) noexcept(is_nothrow_v<BinaryOp>)
    requires (popcount == 1)
    {
        static constexpr auto idx = std::countr_zero(V);
        return permute<idx, idx, idx, idx>(
            static_cast<vector<T>>(std::invoke(op, value, value)));
    }

    template <vector_binary_invocable_for<T> BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) static inline auto
    operator()(vector<T> value, BinaryOp op) noexcept(is_nothrow_v<BinaryOp>) {
        auto const result = [&]<int I>(this auto const self, immediate<I>) {
            if constexpr (I == 1) {
                return value;
            } else {
                static constexpr auto [L, R] =
                    details::reduce::shift_v<popcount>(immediate_v<I>);

                static constexpr packed_indices_for<T> idx{mask_for_v<T, V>, L};
                auto const left = self(immediate_v<L>);
                auto const right = self(immediate_v<R>);
                return [&]<size_t... Is>(std::index_sequence<Is...>) {
                    return static_cast<vector<T>>(
                        std::invoke(op, left, permute<idx[Is]...>(right)));
                }(std::make_index_sequence<element_count_v<T>>{});
            }
        }(popcount);

        return reduction_result<T, V>{result};
    }
};

} // namespace details::reduce

DPL_EXPORT template <std::integral auto V, xmm_element T,
    details::reduce::vector_binary_invocable_for<T> BinaryOp>
requires requires { typename mask_for<T, V>; }
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline reduction_result<T, V>
reduce(vector<T> value, BinaryOp op, mask_for<T, V> = {}) noexcept(
    reduce_t<T, V>::is_nothrow_v<BinaryOp>) {
    static constexpr details::reduce::reduce_t<T, V> impl{};
    return impl(value, op);
}

DPL_EXPORT template <std::integral auto V, xmm_element T>
requires requires { typename mask_for<T, V>; }
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline reduction_result<T, V>
hmin(vector<T> value, mask_for<T, V> = {}) noexcept {
    return reduce<V>(
        value, [](auto left, auto right) noexcept { return min(left, right); });
}

DPL_EXPORT template <std::integral auto V, xmm_element T>
requires requires { typename mask_for<T, V>; }
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline reduction_result<T, V>
hmax(vector<T> value, mask_for<T, V> = {}) noexcept {
    return reduce<V>(
        value, [](auto left, auto right) noexcept { return max(left, right); });
}

namespace details::reduce {

#define DEFINE_REDUCTION_SHIFT(RESULT, LEFT, RIGHT)                        \
    static consteval shift_result operator()(immediate<RESULT>) noexcept { \
        return {LEFT, RIGHT};                                              \
    }                                                                      \
    static_assert(RESULT == LEFT + RIGHT);                                 \
    static_assert(LEFT >= RIGHT)

template <>
struct shift_t<2> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
};

template <>
struct shift_t<3> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
};

template <>
struct shift_t<4> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(4, 2, 2);
};

template <>
struct shift_t<5> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(5, 3, 2);
};
template <>
struct shift_t<6> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(6, 3, 3);
};
template <>
struct shift_t<7> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(4, 2, 2);
    DEFINE_REDUCTION_SHIFT(7, 4, 3);
};
template <>
struct shift_t<8> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(4, 2, 2);
    DEFINE_REDUCTION_SHIFT(8, 4, 4);
};
template <>
struct shift_t<9> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(6, 3, 3);
    DEFINE_REDUCTION_SHIFT(9, 6, 3);
};
template <>
struct shift_t<10> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(5, 3, 2);
    DEFINE_REDUCTION_SHIFT(10, 5, 5);
};
template <>
struct shift_t<11> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(4, 2, 2);
    DEFINE_REDUCTION_SHIFT(7, 4, 3);
    DEFINE_REDUCTION_SHIFT(11, 7, 4);
};
template <>
struct shift_t<12> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(6, 3, 3);
    DEFINE_REDUCTION_SHIFT(12, 6, 6);
};
template <>
struct shift_t<13> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(5, 3, 2);
    DEFINE_REDUCTION_SHIFT(8, 5, 3);
    DEFINE_REDUCTION_SHIFT(13, 8, 5);
};
template <>
struct shift_t<14> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(4, 2, 2);
    DEFINE_REDUCTION_SHIFT(7, 4, 3);
    DEFINE_REDUCTION_SHIFT(14, 7, 7);
};
template <>
struct shift_t<15> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(3, 2, 1);
    DEFINE_REDUCTION_SHIFT(5, 3, 2);
    DEFINE_REDUCTION_SHIFT(10, 5, 5);
    DEFINE_REDUCTION_SHIFT(15, 10, 5);
};
template <>
struct shift_t<16> {
    DEFINE_REDUCTION_SHIFT(2, 1, 1);
    DEFINE_REDUCTION_SHIFT(4, 2, 2);
    DEFINE_REDUCTION_SHIFT(8, 4, 4);
    DEFINE_REDUCTION_SHIFT(16, 8, 8);
};

#undef DEFINE_REDUCTION_SHIFT
} // namespace details::reduce

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END
