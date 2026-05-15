// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/constants/digits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void dot_product(...) noexcept = delete;

template <typename T, typename L, typename R,
    typename A = common_abi_t<L, R, T>>
concept unqualified_dot_product = requires(T acc, L left, R right) {
    {
        dot_product(internal::abi<A>, acc, left, right)
    } -> simd_with<typename T::value_type, A>;
};

struct dot_product_t {
public:
    template <floating_point_simd Result, floating_point_simd L,
        equivalent_simd_as<L> R>
    requires (dx::digits_v<typename Result::value_type> >=
                 dx::digits_v<typename L::value_type>) &&
        same_abi_simd_as<Result, L> &&
        (unqualified_dot_product<Result, L, R> ||
            unqualified_dot_product<basic_type_t<Result>, basic_type_t<L>,
                basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Result acc, L left, R right) noexcept {
        using A = typename Result::abi_type;
        if constexpr (unqualified_dot_product<Result, L, R, A>) {
            return dot_product(internal::abi<A>, acc, left, right);
        } else {
            return dot_product(internal::abi<A>, dx::to_basic_type(acc),
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }
};

template <typename T>
struct zdot_product_t {
public:
    template <floating_point_simd In>
    requires regular_invocable<dot_product_t, rebind_simd_t<In, T>, In, In>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(In left, In right) noexcept
    requires floating_point<T>
    {
        return dot_product_t::operator()(
            dx::zero_v<rebind_simd_t<In, T>>, left, right);
    }

    template <floating_point_simd L, equivalent_simd_as<L> R>
    requires regular_invocable<dot_product_t, L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept
    requires floating_point_simd<T>
    {
        return dx::reinterpret<T>(
            dot_product_t::operator()(dx::zero_v<T>, left, right));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::dot_product_t dot_product{};
DPL_EXPORT template <typename T>
inline constexpr internal::zdot_product_t<T> zdot_product{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
