// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/select.h"
#include "dpl/core/operations/slide.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_order_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t>
void rotate(...) noexcept = delete;
void rotate(...) noexcept = delete;

template <typename L, typename R>
concept unqualified_rotatei = requires(L val) {
    {
        rotate<(R::value % element_count<L>)>(internal::abi<T>, val)
    } -> equivalent_simd_as<L>;
};

template <typename T>
concept unqualified_rotate = requires(T val, size_t lanes) {
    { rotate(internal::abi<T>, val, lanes) } -> equivalent_simd_as<T>;
};

struct rotate_t {
private:
    template <simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto fallback(basic_simd<E, A> val, size_t lanes) noexcept {
        lanes %= element_count<E, A>;
        return dx::slide_right(val, val, element_count<E, A> - lanes);
    }

    template <simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto fallback(basic_simd_mask<E, A> val, size_t lanes) noexcept {
        lanes %= element_count<E, A>;
        return dx::slide_right(val, val, element_count<E, A> - lanes);
    }

    template <size_t V, simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto fallbacki(basic_simd<E, A> val) noexcept {
        return dx::slide_righti<V>(val, val);
    }

    template <size_t V, simd_element E, fixed_width_abi A,
        integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto fallbacki(basic_simd_mask<E, A> val) noexcept {
        return dx::slide_righti<V>(val, val);
    }

public:
    template <fixed_width_class L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R lanes) noexcept {
        static_assert(R::value > 0);
        constexpr auto V = R::value % element_count<L>;
        if constexpr (unqualified_rotatei<L, R>) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return fallbacki<V>(arg);
                } else {
                    return rotate<V>(internal::abi<L>, arg);
                }
            } else {
                return rotate<V>(internal::abi<L>, arg);
            }
        } else if constexpr (basic_simd_class<L>) {
            return fallbacki<V>(arg);
        } else {
            return operator()(dx::to_basic_type(arg), lanes);
        }
    }

    template <scalable_class L, integral_constant_like R>
    requires unqualified_rotatei<L, R> ||
        unqualified_rotatei<basic_type_t<L>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, R lanes) noexcept {
        static_assert(R::value > 0);
        constexpr auto V = R::value % element_count<L>;
        if constexpr (unqualified_rotatei<L, R>) {
            return rotate<V>(internal::abi<L>, arg);
        } else {
            return rotate<V>(internal::abi<L>, dx::to_basic_type(arg));
        }
    }

    template <fixed_width_class L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t lanes) noexcept {
        if constexpr (unqualified_rotate<L>) {
            if constexpr (basic_simd_class<L>) {
                if consteval {
                    return fallback(arg, lanes);
                } else {
                    return rotate(internal::abi<L>, arg, lanes);
                }
            } else {
                return rotate(internal::abi<L>, arg, lanes);
            }
        } else if constexpr (basic_simd_class<L>) {
            return fallback(arg, lanes);
        } else {
            return operator()(dx::to_basic_type(arg), lanes);
        }
    }

    template <scalable_class L>
    requires unqualified_rotate<L> || unqualified_rotate<basic_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L arg, size_t lanes) noexcept {
        if constexpr (unqualified_rotate<L>) {
            return rotate(internal::abi<L>, arg, lanes);
        } else {
            return rotate(internal::abi<L>, dx::to_basic_type(arg), lanes);
        }
    }
};

template <size_t V>
struct rotatei_t {
public:
    template <simd_class T>
    requires requires regular_invocable<rotate_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return rotate_t::operator()(arg, imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <size_t V>
inline constexpr internal::rotatei_t<V> rotatei{};
DPL_EXPORT inline constexpr internal::rotate_t rotate{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
