// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/common_order_type.h"
#  include "dpl/core/type_traits/to_simd_mask_type.h"
#  include "dpl/std/type_traits/is_invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpeq(...) noexcept = delete;
void cmpneq(...) noexcept = delete;
void cmplt(...) noexcept = delete;
void cmple(...) noexcept = delete;
void cmpgt(...) noexcept = delete;
void cmpge(...) noexcept = delete;

template <typename L, typename R>
using compare_result DPL_NODEBUG =
    to_simd_mask_type_t<common_order_simd_t<L, R>>;

struct cmpeq_t : binary_operation_base<cmpeq_t> {
private:
    template <simd_element L, common_order_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            lhs, rhs, [](auto lhs, auto rhs) -> bool { return lhs == rhs; });
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmpeq(internal::abi<A>, lhs, rhs); }) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmpeq(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        return cmpeq(internal::abi<A>, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmpeq(internal::abi<A>, lhs, rhs); }) {
            return cmpeq(internal::abi<A>, lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<cmpeq_t>::operator();
};

struct cmpneq_t : binary_operation_base<cmpneq_t> {
private:
    template <simd_element L, common_order_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            lhs, rhs, [](auto lhs, auto rhs) -> bool { return lhs != rhs; });
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmpneq(internal::abi<A>, lhs, rhs); }) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmpneq(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        return cmpneq(internal::abi<A>, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmpneq(internal::abi<A>, lhs, rhs); }) {
            return cmpneq(internal::abi<A>, lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<cmpneq_t>::operator();
};

struct cmplt_t : binary_operation_base<cmplt_t> {
private:
    template <simd_element L, common_order_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            lhs, rhs, [](auto lhs, auto rhs) -> bool { return lhs < rhs; });
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmplt(internal::abi<A>, lhs, rhs); }) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmplt(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        return cmplt(internal::abi<A>, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmplt(internal::abi<A>, lhs, rhs); }) {
            return cmplt(internal::abi<A>, lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<cmplt_t>::operator();
};

struct cmple_t : binary_operation_base<cmple_t> {
private:
    template <simd_element L, common_order_with<L> R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(
        basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_simd_mask<T, A>>(
            lhs, rhs, [](auto lhs, auto rhs) -> bool { return lhs <= rhs; });
    }

public:
    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmple(internal::abi<A>, lhs, rhs); }) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmple(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <basic_simd_type L, basic_simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        return cmple(internal::abi<A>, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmple(internal::abi<A>, lhs, rhs); }) {
            return cmple(internal::abi<A>, lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<cmple_t>::operator();
};

struct cmpgt_t : binary_operation_base<cmpgt_t> {
public:
    template <simd_type L, simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmpgt(internal::abi<A>, lhs, rhs); }) {
            return cmpgt(internal::abi<A>, lhs, rhs);
        } else {
            return cmplt_t::operator()(rhs, lhs);
        }
    }

    using binary_operation_base<cmpgt_t>::operator();
};

struct cmpge_t : binary_operation_base<cmpge_t> {
public:
    template <simd_type L, simd_type R>
    requires common_order_simd_with<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> common_order_simd_with<compare_result<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { cmpge(internal::abi<A>, lhs, rhs); }) {
            return cmpge(internal::abi<A>, lhs, rhs);
        } else {
            return cmple_t::operator()(rhs, lhs);
        }
    }

    using binary_operation_base<cmpge_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cmpeq_t cmpeq{};
DPL_EXPORT inline constexpr internal::cmpneq_t cmpneq{};
DPL_EXPORT inline constexpr internal::cmplt_t cmplt{};
DPL_EXPORT inline constexpr internal::cmple_t cmple{};
DPL_EXPORT inline constexpr internal::cmpgt_t cmpgt{};
DPL_EXPORT inline constexpr internal::cmpge_t cmpge{};
} // namespace cpo

DPL_EXPORT template <typename D>
class comparison_simd_interface {
public:
    template <typename R>
    requires regular_invocable<internal::cmpeq_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpeq_t, D, R> operator==(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpeq(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmpneq_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpneq_t, D, R> operator!=(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpneq(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmplt_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmplt_t, D, R> operator<(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmplt(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmple_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmple_t, D, R> operator<=(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmple(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmpgt_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpgt_t, D, R> operator>(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpgt(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmpge_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpge_t, D, R> operator>=(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpge(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpeq_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpeq_t, L, D> operator==(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpeq(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpneq_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpneq_t, L, D> operator!=(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpneq(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmplt_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmplt_t, L, D> operator<(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmplt(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmple_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmple_t, L, D> operator<=(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmple(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpgt_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpgt_t, L, D> operator>(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpgt(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpge_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpge_t, L, D> operator>=(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpge(lhs, rhs);
    }
};

/**
 * The following are not exported by design
 */

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpeq_t, L, R> operator==(
    L lhs, R rhs) noexcept {
    return datapar::cmpeq(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpneq_t, L, R> operator!=(
    L lhs, R rhs) noexcept {
    return datapar::cmpneq(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmplt_t, L, R> operator<(
    L lhs, R rhs) noexcept {
    return datapar::cmplt(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmple_t, L, R> operator<=(
    L lhs, R rhs) noexcept {
    return datapar::cmple(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpgt_t, L, R> operator>(
    L lhs, R rhs) noexcept {
    return datapar::cmpgt(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpge_t, L, R> operator>=(
    L lhs, R rhs) noexcept {
    return datapar::cmpge(lhs, rhs);
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
