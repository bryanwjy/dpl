// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <auto>
void splice(...) noexcept = delete;
void splice(...) noexcept = delete;

struct splice_t;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_splice = requires(M mask, L lhs, R rhs) {
    {
        splice(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_select_vector<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_splice = requires(M mask, L lhs, R rhs) {
    { splice(mask, lhs, rhs) } -> extended_select_vector<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_splice = unqualified_extended_splice<M, L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        decayable_mask_for<M, operation_category::lane_permutation> &&
        regular_invocable<splice_t, canonical_type_t<M>, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_splicei = requires(L lhs, R rhs) {
    {
        splice<internal::select_mask<M, L, R>()()>(internal::abi<A>, lhs, rhs)
    } -> canonical_selecti_vector<L, R, A>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_splicei = requires(L lhs, R rhs) {
    {
        splice<internal::select_mask<M, L, R>()()>(lhs, rhs)
    } -> extended_selecti_vector<L, R, A>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_splicei = unqualified_extended_splicei<M, L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<splice_t, M, canonical_type_t<L>,
            canonical_type_t<R>>);

struct splice_t {
private:
    template <typename EM, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_mask<EM, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
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

    template <typename L, typename R>
    static consteval auto selective_abi() noexcept {
        if constexpr (simd_class<L>) {
            return typename L::abi_type{};
        } else {
            return typename R::abi_type{};
        }
    }

    template <simd_class L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<L> const& arg) noexcept {
        return (arg);
    }

    template <typename L, simd_class R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<R> const& arg) noexcept {
        return (arg);
    }

    template <typename L, simd_class R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<L> arg) noexcept {
        return dx::broadcast<R>(arg);
    }

    template <simd_class L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<R> arg) noexcept {
        return dx::broadcast<L>(arg);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> ME, simd_element_for<A> E>
    requires common_size_with<ME, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> tval, basic_vector<E, A> fval) noexcept {
        if constexpr (unqualified_canonical_select<basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>, A>) {
            if consteval {
                return fallback(mask, tval, fval);
            } else {
                return splice(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return fallback(mask, tval, fval);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi LA, simd_abi RA,
        simd_element_for<LA> E>
    requires simd_element_for<E, RA> && common_size_with<ME, E> &&
        (different_from<MA, LA> || different_from<MA, RA> ||
            different_from<LA, RA> || scalable_abi<MA> || scalable_abi<LA> ||
            scalable_abi<RA>) &&
        unqualified_canonical_splice<basic_mask<ME, MA>, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA, MA>> operator()(
        basic_mask<ME, MA> mask, basic_vector<E, LA> tval,
        basic_vector<E, RA> fval) noexcept {
        using A = common_abi_t<LA, RA, MA>;
        return splice(internal::abi<A>, mask, tval, fval);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        common_size_with<typename M::value_type, typename R::value_type> &&
        common_size_with<typename M::value_type, typename L::value_type> &&
        unqualified_splice<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L tval, R fval) noexcept {
        if constexpr (unqualified_extended_splice<M, L, R>) {
            return splice(mask, tval, fval);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        constexpr auto V =
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask)();
        if constexpr (unqualified_canonical_splicei<M, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallbacki<V>(lhs, rhs);
            } else {
                return splice<V>(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallbacki<V>(lhs, rhs);
        }
    }

    template <typename M, fixed_width_abi LA, fixed_width_abi RA,
        simd_element_for<LA> E>
    requires simd_element_for<E, RA> && different_from<LA, RA> &&
        const_mask_for<M, basic_vector<E, LA>> &&
        const_mask_for<M, basic_vector<E, RA>> &&
        unqualified_canonical_splicei<M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        using L = basic_vector<E, LA>;
        using R = basic_vector<E, RA>;
        constexpr auto V = internal::select_mask<M, L, R>()();
        return splice<V>(internal::abi<A>, mask, lhs, rhs);
    }

    template <const_mask_like M, fixed_width_vector L, fixed_width_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        unqualified_splicei<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_splicei<M, L, R>) {
            constexpr auto V = internal::select_mask<M, L, R>()();
            return splice<V>(lhs, rhs);
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, typename L, typename R>
    requires (simd_class<L> && !simd_class<R> && broadcastable_to<R, L>) ||
        (simd_class<R> && !simd_class<L> && broadcastable_to<L, R>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(M mask, L lhs, R rhs) noexcept
    requires requires {
        splice_t::operator()(mask, splice_t::selective_cast<L, R>(lhs),
            splice_t::selective_cast<L, R>(rhs));
    }
    {
        using A =
            common_abi_t<typename M::abi_type, decltype(selective_abi<L, R>())>;
        if constexpr (canonical_mask<M> &&
            (canonical_class<L> || canonical_class<R>)) {
            if constexpr (requires {
                              splice(internal::abi<A>, mask, lhs, rhs);
                          }) {
                if consteval {
                    return operator()(mask, splice_t::selective_cast<L, R>(lhs),
                        splice_t::selective_cast<L, R>(rhs));
                } else {
                    return splice(internal::abi<A>, mask, lhs, rhs);
                }
            } else {
                return operator()(mask, splice_t::selective_cast<L, R>(lhs),
                    splice_t::selective_cast<L, R>(rhs));
            }
        } else if constexpr (simd_class<L> || simd_class<R>) {
            if constexpr (requires { splice(mask, lhs, rhs); }) {
                return splice(mask, lhs, rhs);
            } else if constexpr (simd_class<L>) {
                return operator()(mask, dx::to_canonical(lhs), rhs);
            } else {
                return operator()(mask, lhs, dx::to_canonical(rhs));
            }
        }
    }

    template <typename M, typename L, typename R>
    requires (simd_class<L> && !simd_class<R> && broadcastable_to<R, L> &&
                 const_mask_for<M, L>) ||
        (simd_class<R> && !simd_class<L> && broadcastable_to<L, R> &&
            const_mask_for<M, R>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(M mask, L lhs, R rhs) noexcept
    requires requires {
        splice_t::operator()(mask, splice_t::selective_cast<L, R>(lhs),
            splice_t::selective_cast<L, R>(rhs));
    }
    {
        constexpr auto V = [](M mask) {
            if constexpr (simd_class<L>) {
                return dx::to_compatible_const_mask<L>(mask)();
            } else {
                return dx::to_compatible_const_mask<R>(mask)();
            }
        }(mask);

        using A = decltype(selective_abi<L, R>());
        if constexpr ((canonical_class<L> || canonical_class<R>)) {
            if constexpr (requires { splice<V>(internal::abi<A>, lhs, rhs); }) {
                if consteval {
                    return operator()(mask, splice_t::selective_cast<L, R>(lhs),
                        splice_t::selective_cast<L, R>(rhs));
                } else {
                    return splice<V>(internal::abi<A>, lhs, rhs);
                }
            } else {
                return operator()(mask, splice_t::selective_cast<L, R>(lhs),
                    splice_t::selective_cast<L, R>(rhs));
            }
        } else if constexpr (requires { splice<V>(lhs, rhs); }) {
            return splice<V>(lhs, rhs);
        } else if constexpr (simd_class<L>) {
            return operator()(mask, dx::to_canonical(lhs), rhs);
        } else {
            return operator()(mask, lhs, dx::to_canonical(rhs));
        }
    }
};

template <auto V>
struct splicei_t {};

template <integral auto V>
struct splicei_t<V> {
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

    template <simd_class L, typename R>
    requires (!simd_class<R>) && requires { typename mask_type<L>; } &&
        regular_invocable<splice_t, mask_type<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<L> mask{};
        return splice_t::operator()(mask, lhs, rhs);
    }

    template <typename L, simd_class R>
    requires (!simd_class<L>) && requires { typename mask_type<R>; } &&
        regular_invocable<splice_t, mask_type<R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<R> mask{};
        return splice_t::operator()(mask, lhs, rhs);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
DPL_EXPORT inline constexpr internal::splice_t splice{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
