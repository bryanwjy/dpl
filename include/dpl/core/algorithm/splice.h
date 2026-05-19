// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/common_order_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <auto>
void splice(...) noexcept = delete;
void splice(...) noexcept = delete;

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_splice = requires(C cond, L lhs, R rhs) {
    { splice(internal::abi<A>, cond, lhs, rhs) } -> ternary_result_of<L, R, A>;
};

template <typename C, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_splicei = const_mask_for<C, L> && requires(L lhs, R rhs) {
    {
        splice<const_mask_v<L, C>>(internal::abi<A>, lhs, rhs)
    } -> ternary_result_of<L, R, A>;
};

struct splice_t {
private:
    template <typename EM, typename EL, typename ER, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_mask<EM, A> mask,
        basic_vector<EL, A> lhs, basic_vector<ER, A> rhs) noexcept {
        using mask_type = basic_mask<EM, A>;
        auto const low = dx::countr_zero(mask);
        auto const high = dx::countl_zero(mask);
        return dx::slide_left(dx::shift_right(lhs, high), rhs, high + low);
    }

    template <auto V, typename EL, typename ER, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<EL, A> lhs, basic_vector<ER, A> rhs) noexcept {
        constexpr const_mask<simd_abi_traits<EL, A>::size, V> inmask{};
        constexpr auto low = dx::countr_zero(inmask);
        constexpr auto high = dx::countr_zero(inmask);
        return dx::slide_left(
            dx::shift_right(lhs, imm<high>), rhs, imm<high + low>);
    }

public:
    template <simd_vector L, simd_mask M, selectable_with<L, M> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = typename M::abi_type;
        if constexpr (unqualified_splice<M, L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R> &&
                canonical_mask<M>) {
                if consteval {
                    return fallback(mask, lhs, rhs);
                } else {
                    return splice(internal::abi<A>, mask, lhs, rhs);
                }
            } else {
                return splice(internal::abi<A>, mask, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R> &&
            canonical_mask<M>) {
            return fallback(mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_mask M, selectable_with<L, M> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_splice<M, L, R> ||
            unqualified_splice<M, canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_splice<M, L, R>) {
            return splice(internal::abi<A>, mask, lhs, rhs);
        } else {
            return splice(internal::abi<A>, dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, selectablei_with<L> R, const_mask_for<L> M>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        constexpr auto V =
            decltype(dx::to_compatible_const_mask<L>(mask))::value;
        if constexpr (unqualified_splicei<M, L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    return fallbacki<V>(lhs, rhs);
                } else {
                    return splice<V>(internal::abi<A>, lhs, rhs);
                }
            } else {
                return splice<V>(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallbacki<V>(lhs, rhs);
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, selectablei_with<L> R, const_mask_for<L> M>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_splicei<M, L, R> ||
            unqualified_splicei<M, canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        constexpr auto V =
            decltype(dx::to_compatible_const_mask<L>(mask))::value;
        if constexpr (unqualified_splicei<M, L, R>) {
            return splice<V>(internal::abi<A>, lhs, rhs);
        } else {
            return splice<V>(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
};

template <auto V>
struct splicei_t {};

template <integral auto V>
struct splicei_t<V> : binary_operation_base<splicei_t<V>> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <simd_vector L, simd_vector R>
    requires requires {
        typename mask_type<L>;
        requires regular_invocable<splice_t, mask_type<L>, L, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<L> mask{};
        return splice_t::operator()(mask, lhs, rhs);
    }

    using binary_operation_base<splicei_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
DPL_EXPORT inline constexpr internal::splice_t splice{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
