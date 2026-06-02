// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"
#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic/add.h"
#  include "dpl/core/operations/broadcast_lane.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/apply.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct scan_base {
protected:
    template <integral_constant_like Imm, fixed_width_vector T,
        typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL inclusive(
        Imm, T val, BinaryOp&& op) noexcept {

        [&]<size_t I>(this auto self, immediate<I> offset) {
            using bitset_t = bitset<simd_abi_traits<T>::size>;
            constexpr auto lane_mask = __DPL apply(
                [offset]<typename... Is>(Is... idx) {
                    constexpr auto S = simd_abi_traits<T>::size();
                    constexpr auto set =
                        bitset_t((Is::value >= I && Is::value < Imm::value)...);
                    using underlying = typename bitset_t::underlying_type;
                    if constexpr (integral<underlying>) {
                        return const_mask<S, static_cast<underlying>(set)>{};
                    } else {
                        return const_mask<S, set>{};
                    }
                },
                iota_sequence<T>);

            auto const shifted = dx::shift_right(val, offset);
            val = dx::select(lane_mask, __DPL invoke(op, shifted, val), val);
            if constexpr (I < Imm::value) {
                self(imm<I * 2>);
            }
        }(imm<1zu>);

        return val;
    }

    template <simd_vector T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL inclusive(
        size_t size, T val, BinaryOp&& op) noexcept {
        auto const idx = dx::lane_index<T>();
        for (auto i = 0zu; i < size; i <<= 1) {
            auto const shifted = dx::shift_right(val, i);
            using sint = signed_representation_t<typename T::value_type>;
            auto const offset = dx::broadcast_lane(idx, i);
            val = dx::select(idx >= offset, op(shifted, val), val);
        }

        return val;
    }

    template <simd_vector T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL inclusive(
        T val, BinaryOp&& op) noexcept {
        if constexpr (fixed_width_vector<T>) {
            constexpr auto N = simd_abi_traits<T>::size();
            return scan_base::inclusive(
                imm<N>, val, __DPL forward<BinaryOp>(op));
        } else {
            return scan_base::inclusive(
                simd_abi_traits<T>::size(), val, __DPL forward<BinaryOp>(op));
        }
    }

    template <integral_constant_like Imm, fixed_width_vector T,
        typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL exclusive(
        Imm size, T val, T init, BinaryOp&& op) noexcept {
        val = scan_base::inclusive(size, val, op);
        auto const result =
            dx::select(imm<0b1>, init, dx::shift_right(val, imm<1zu>));
        return dx::select(imm<0b1>, result, op(init, result));
    }

    template <simd_vector T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL exclusive(
        size_t size, T val, T init, BinaryOp&& op) noexcept {
        val = scan_base::inclusive(size, val, op);
        if constexpr (fixed_width_vector<T>) {
            auto const result =
                dx::select(imm<0b1>, init, dx::shift_right(val, imm<1zu>));
            return dx::select(imm<0b1>, result, op(init, result));
        } else {
            auto const iszero = dx::lane_index<T>() == 0;
            auto const result =
                dx::select(iszero, init, dx::shift_right(val, 1zu));
            return dx::select(iszero, result, op(init, result));
        }
    }

    template <simd_vector T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL exclusive(
        T val, T init, BinaryOp&& op) noexcept {
        if constexpr (fixed_width_vector<T>) {
            constexpr auto N = simd_abi_traits<T>::size();
            return scan_base::exclusive(
                imm<N>, val, init, __DPL forward<BinaryOp>(op));
        } else {
            return scan_base::exclusive(simd_abi_traits<T>::size(), val, init,
                __DPL forward<BinaryOp>(op));
        }
    }
};

struct exscan_sum_base;

template <typename T, typename I>
concept unqualified_canonical_exscan_sum = requires(T val, I init) {
    {
        exscan_sum(internal::abi<T>, val, init)
    } -> canonical_arithmetic_result<T>;
};

template <typename T, typename I>
concept unqualified_extended_exscan_sum = requires(T val, I init) {
    { exscan_sum(val, init) } -> extended_arithmetic_result<T>;
};

template <typename T, typename M>
concept extended_mask_scan_result =
    simd_mask<M> && simd_vector<T> && signed_integral<typename T::value_type> &&
    common_size_with<typename T::value_type, simd_lane_type_t<M>> &&
    common_abi_with<typename T::abi_type, typename M::abi_type>;

template <typename T, typename M>
concept canonical_mask_scan_result = extended_mask_scan_result<T, M> &&
    same_as<typename T::abi_type, typename M::abi_type>;

template <typename T>
concept unqualified_canonical_mask_scan = requires(T val) {
    { exscan_sum(internal::abi<T>, val) } -> canonical_mask_scan_result<T>;
};

template <typename T>
concept unqualified_extended_mask_scan = requires(T val) {
    { exscan_sum(val) } -> extended_mask_scan_result<T>;
};

struct exscan_sum_base : protected scan_base {
    template <canonical_vector T, broadcastable_to<T> I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I init) noexcept {
        if constexpr (unqualified_canonical_exscan_sum<T, I>) {
            if consteval {
                return scan_base::exclusive(
                    val, dx::broadcast<T>(init), dx::add);
            } else {
                return exscan_sum(internal::abi<T>, val, init);
            }
        } else {
            return scan_base::exclusive(val, dx::broadcast<T>(init), dx::add);
        }
    }

    template <extended_vector T, broadcastable_to<T> I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I init) noexcept {
        if constexpr (unqualified_extended_exscan_sum<T, I>) {
            return exscan_sum(val, init);
        } else {
            return scan_base::exclusive(val, dx::broadcast<T>(init), dx::add);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<E, A> mask) noexcept {
        using sint = signed_representation_t<E>;
        if constexpr (unqualified_canonical_mask_scan<basic_mask<E, A>>) {
            if consteval {
                using idx_type = decltype(dx::lane_index<E, A>());
                auto const vec = dx::select(
                    mask, dx::broadcast<idx_type>(dx::one), dx::zero);
                return operator()(vec, dx::zero);
            } else {
                return exscan_sum(internal::abi<A>, mask);
            }
        } else {
            using idx_type = decltype(dx::lane_index<E, A>());
            auto const vec =
                dx::select(mask, dx::broadcast<idx_type>(dx::one), dx::zero);
            return operator()(vec, dx::zero);
        }
    }

    template <extended_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T mask) noexcept {
        if constexpr (unqualified_extended_mask_scan<T>) {
            return exscan_sum(mask);
        } else {
            using idx_type = decltype(dx::lane_index<T>());
            auto const vec =
                dx::select(mask, dx::broadcast<idx_type>(dx::one), dx::zero);
            return operator()(vec, dx::zero);
        }
    }
};

struct scan_sum_base;

template <typename T>
concept unqualified_canonical_scan_sum = requires(T val) {
    { scan_sum(internal::abi<T>, val) } -> canonical_arithmetic_result<T>;
};

template <typename T>
concept unqualified_extended_scan_sum = requires(T val) {
    { scan_sum(val) } -> extended_arithmetic_result<T>;
};

struct scan_sum_base : protected scan_base {
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_scan_sum<basic_vector<E, A>>) {
            if consteval {
                return scan_base::inclusive(val, dx::add);
            } else {
                return scan_sum(internal::abi<A>, val);
            }
        } else {
            return scan_base::inclusive(val, dx::add);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_scan_sum<T>) {
            return scan_sum(val);
        } else {
            return scan_base::inclusive(val, dx::add);
        }
    }
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
