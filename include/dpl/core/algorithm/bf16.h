// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/inner_product.h"

#if !DPL_MODULES
#  include "dpl/core/basic/to_basic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void dot_product(...) noexcept = delete;

template <typename F32, typename BF16, typename A = common_abi_t<F32, BF16>>
concept unqualified_dot_product = requires(
    F32 f32, BF16 bf16) { dot_product(internal::abi<A>, f32, bf16, bf16); };

struct dot_product_t {
private:
    template <brain_float E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<float, A> DPL_VECTORCALL
        fallback(basic_simd<float, A> acc, basic_simd<E, A> left,
            basic_simd<E, A> right) noexcept {
        return acc + dx::inner_product(left, right);
    }

public:
    template <floating_point_simd F32, floating_point_simd BF16>
    requires common_float_with<typename F32::value_type, float> &&
        brain_float<typename BF16::value_type> && same_abi_simd_as<F32, BF16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr F32 operator()(F32 acc, BF16 left, BF16 right) noexcept {
        using A = typename F32::abi_type;
        if constexpr (unqualified_dot_product<F32, BF16, A>) {
            if constexpr (basic_simd_type<F32> && basic_simd_type<BF16>) {
                if consteval {
                    return fallback(acc, left, right);
                } else {
                    return dot_product(internal::abi<F32>, acc, left, right);
                }
            } else {
                return dot_product(internal::abi<F32>, acc, left, right);
            }
        } else if constexpr (basic_simd_type<F32> && basic_simd_type<BF16>) {
            return fallback(acc, left, right);
        } else {
            return fallback(dx::to_basic_type(acc), dx::to_basic_type(left),
                dx::to_basic_type(right));
        }
    }

    template <floating_point_simd F32, floating_point_simd BF16>
    requires common_float_with<typename F32::value_type, float> &&
        brain_float<typename BF16::value_type> &&
        (!same_abi_simd_as<F32, BF16>) &&
        (unqualified_dot_product<F32, BF16> ||
            unqualified_dot_product<basic_type_t<F32>, basic_type_t<BF16>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr F32 operator()(F32 acc, BF16 left, BF16 right) noexcept {
        using A = common_abi_t<F32, BF16>;
        if constexpr (unqualified_dot_product<F32, BF16>) {
            return dot_product(internal::abi<A>, acc, left, right);
        } else {
            return dot_product(internal::abi<A>, dx::to_basic_type(acc),
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }
};

} // namespace datapar::internal

namespace datapar::bf16 {
DPL_EXPORT inline constexpr internal::dot_product_t dot_product{};
}
DPL_DEFAULT_NAMESPACE_END
