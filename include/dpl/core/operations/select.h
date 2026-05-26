// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/const_mask.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/all_bits.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void select(...) noexcept = delete;
template <auto>
void select(...) noexcept = delete;

struct select_t;

template <typename T, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept canonical_select_vector =
    simd_vector<T> && same_as<typename L::value_type, typename T::value_type> &&
    same_as<typename R::value_type, typename T::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename L::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename R::value_type> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_select_vector =
    simd_vector<T> && same_as<typename L::value_type, typename T::value_type> &&
    same_as<typename R::value_type, typename T::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename L::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename R::value_type> &&
    common_abi_with<A, typename T::abi_type>;

template <typename T, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept canonical_select_mask = simd_mask<T> &&
    common_size_with<typename L::value_type, typename T::value_type> &&
    common_size_with<typename R::value_type, typename T::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename L::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename R::value_type> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_select_mask = simd_mask<T> &&
    common_size_with<typename L::value_type, typename T::value_type> &&
    common_size_with<typename R::value_type, typename T::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename L::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename R::value_type> &&
    common_abi_with<A, typename T::abi_type>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_select = requires(M mask, L lhs, R rhs) {
    {
        select(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_select_vector<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_select = requires(M mask, L lhs, R rhs) {
    { select(mask, lhs, rhs) } -> extended_select_vector<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_select = unqualified_extended_select<M, L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        decayable_mask_for<M, operation_category::lane_agnostic> &&
        regular_invocable<select_t, canonical_type_t<M>, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mask_select = requires(M mask, L lhs, R rhs) {
    {
        select(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_select_mask<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mask_select = requires(M mask, L lhs, R rhs) {
    { select(mask, lhs, rhs) } -> extended_select_mask<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_mask_select =
    unqualified_canonical_mask_select<M, L, R, A> ||
    unqualified_extended_mask_select<M, L, R, A> ||
    (decayable_mask_for<L, operation_category::lane_agnostic> &&
        decayable_mask_for<R, operation_category::lane_agnostic> &&
        decayable_mask_for<M, operation_category::lane_agnostic> &&
        regular_invocable<select_t, canonical_type_t<M>, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename T, typename M, typename R, typename A = common_abi_t<R, M>>
concept canonical_masked_vector =
    simd_vector<T> && same_as<typename R::value_type, typename T::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename R::value_type> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename M, typename R, typename A = common_abi_t<R, M>>
concept extended_masked_vector =
    simd_vector<T> && same_as<typename R::value_type, typename T::value_type> &&
    common_size_with<simd_lane_type_t<M>, typename R::value_type> &&
    common_abi_with<A, typename T::abi_type>;

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_canonical_bitkeep = requires(M mask, R val) {
    {
        select(internal::abi<A>, mask, val, dx::zero)
    } -> canonical_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_extended_bitkeep = requires(M mask, R val) {
    { select(mask, val, dx::zero) } -> extended_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_bitkeep = unqualified_extended_bitkeep<M, R, A> ||
    (decayable_vector_for<R, operation_category::lane_agnostic> &&
        decayable_mask_for<M, operation_category::lane_agnostic> &&
        regular_invocable<select_t, canonical_type_t<M>, canonical_type_t<R>,
            dx::zero_t>);

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_canonical_bitdrop = requires(M mask, R val) {
    {
        select(internal::abi<A>, mask, dx::zero, val)
    } -> canonical_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_extended_bitdrop = requires(M mask, R val) {
    { select(mask, dx::zero, val) } -> extended_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_bitdrop = unqualified_extended_bitdrop<M, R, A> ||
    (decayable_vector_for<R, operation_category::lane_agnostic> &&
        decayable_mask_for<M, operation_category::lane_agnostic> &&
        regular_invocable<select_t, canonical_type_t<M>, dx::zero_t,
            canonical_type_t<R>>);

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_canonical_bitfill = requires(M mask, R val) {
    {
        select(internal::abi<A>, mask, dx::all_bits, val)
    } -> canonical_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_extended_bitfill = requires(M mask, R val) {
    { select(mask, dx::all_bits, val) } -> extended_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_bitfill = unqualified_extended_bitfill<M, R, A> ||
    (decayable_vector_for<R, operation_category::lane_agnostic> &&
        decayable_mask_for<M, operation_category::lane_agnostic> &&
        regular_invocable<select_t, canonical_type_t<M>, dx::all_bits_t,
            canonical_type_t<R>>);

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_canonical_bitspill = requires(M mask, R val) {
    {
        select(internal::abi<A>, mask, val, dx::all_bits)
    } -> canonical_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_extended_bitspill = requires(M mask, R val) {
    { select(mask, val, dx::all_bits) } -> extended_masked_vector<M, R, A>;
};

template <typename M, typename R, typename A = common_abi_t<R, M>>
concept unqualified_bitspill = unqualified_extended_bitspill<M, R, A> ||
    (decayable_vector_for<R, operation_category::lane_agnostic> &&
        decayable_mask_for<M, operation_category::lane_agnostic> &&
        regular_invocable<select_t, canonical_type_t<M>, canonical_type_t<R>,
            dx::all_bits_t>);

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_selecti_vector =
    simd_vector<T> && same_as<typename L::value_type, typename T::value_type> &&
    same_as<typename R::value_type, typename T::value_type> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_selecti_vector =
    simd_vector<T> && same_as<typename L::value_type, typename T::value_type> &&
    same_as<typename R::value_type, typename T::value_type> &&
    common_abi_with<A, typename T::abi_type>;

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_selecti_mask = simd_mask<T> &&
    common_size_with<typename L::value_type, typename T::value_type> &&
    common_size_with<typename R::value_type, typename T::value_type> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_selecti_mask = simd_mask<T> &&
    common_size_with<typename L::value_type, typename T::value_type> &&
    common_size_with<typename R::value_type, typename T::value_type> &&
    common_abi_with<A, typename T::abi_type>;

template <typename M, typename L, typename R>
consteval auto select_mask() noexcept {
    using LM = make_const_mask_t<L, M::value>;
    using RM = make_const_mask_t<R, M::value>;
    if constexpr (LM::width > RM::width) {
        return LM{};
    } else {
        return RM{};
    }
}

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_selecti = requires(L lhs, R rhs) {
    {
        select<internal::select_mask<M, L, R>()()>(internal::abi<A>, lhs, rhs)
    } -> canonical_selecti_vector<L, R, A>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_selecti = requires(L lhs, R rhs) {
    {
        select<internal::select_mask<M, L, R>()()>(lhs, rhs)
    } -> extended_selecti_vector<L, R, A>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_selecti = unqualified_extended_selecti<M, L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<select_t, M, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_mask_selecti = requires(L lhs, R rhs) {
    {
        select<internal::select_mask<M, L, R>()()>(internal::abi<A>, lhs, rhs)
    } -> canonical_selecti_mask<L, R, A>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_mask_selecti = requires(L lhs, R rhs) {
    {
        select<internal::select_mask<M, L, R>()()>(lhs, rhs)
    } -> extended_selecti_mask<L, R, A>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mask_selecti =
    unqualified_canonical_mask_selecti<M, L, R, A> ||
    unqualified_extended_mask_selecti<M, L, R, A> ||
    (decayable_mask_for<L, operation_category::lane_agnostic> &&
        decayable_mask_for<R, operation_category::lane_agnostic> &&
        regular_invocable<select_t, M, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename M, typename R>
concept unqualified_canonical_bitkeepi = requires(R val) {
    {
        select<const_mask_v<R, M>>(internal::abi<R>, val, dx::zero)
    } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_extended_bitkeepi = requires(R val) {
    { select<const_mask_v<R, M>>(val, dx::zero) } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_bitkeepi = unqualified_extended_bitkeepi<M, R> ||
    (decayable_simd_for<R, operation_category::lane_agnostic> &&
        regular_invocable<select_t, M, canonical_type_t<R>, dx::zero_t>);

template <typename M, typename R>
concept unqualified_canonical_bitdropi = requires(R val) {
    {
        select<const_mask_v<R, M>>(internal::abi<R>, dx::zero, val)
    } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_extended_bitdropi = requires(R val) {
    { select<const_mask_v<R, M>>(dx::zero, val) } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_bitdropi = unqualified_extended_bitdropi<M, R> ||
    (decayable_simd_for<R, operation_category::lane_agnostic> &&
        regular_invocable<select_t, M, dx::zero_t, canonical_type_t<R>>);

template <typename M, typename R>
concept unqualified_canonical_bitfilli = requires(R val) {
    {
        select<const_mask_v<R, M>>(internal::abi<R>, dx::all_bits, val)
    } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_extended_bitfilli = requires(R val) {
    { select<const_mask_v<R, M>>(dx::all_bits, val) } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_bitfilli = unqualified_extended_bitfilli<M, R> ||
    (decayable_simd_for<R, operation_category::lane_agnostic> &&
        regular_invocable<select_t, M, dx::all_bits_t, canonical_type_t<R>>);

template <typename M, typename R>
concept unqualified_canonical_bitspilli = requires(R val) {
    {
        select<const_mask_v<R, M>>(internal::abi<R>, val, dx::all_bits)
    } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_extended_bitspilli = requires(R val) {
    { select<const_mask_v<R, M>>(val, dx::all_bits) } -> equivalent_simd_as<R>;
};

template <typename M, typename R>
concept unqualified_bitspilli = unqualified_extended_bitspilli<M, R> ||
    (decayable_simd_for<R, operation_category::lane_agnostic> &&
        regular_invocable<select_t, M, canonical_type_t<R>, dx::all_bits_t>);

struct select_t {
private:
    template <typename M, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_mask<M, A> mask,
        basic_vector<E, A> tval, basic_vector<E, A> fval) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](bool cond, E tval, E fval) { return cond ? tval : fval; }, mask,
            tval, fval);
    }

    template <typename M, typename LE, typename RE, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_mask<M, A> mask,
        basic_mask<LE, A> tval, basic_mask<RE, A> fval) noexcept {
        using E = common_size_type_t<M, LE, RE>;
        return internal::transform<basic_mask<E, A>>(
            [](bool cond, bool tval, bool fval) { return cond ? tval : fval; },
            mask, tval, fval);
    }

    template <auto V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> tval, basic_vector<E, A> fval) noexcept {
        static constexpr const_mask<simd_abi_traits<E, A>::size, V> mask{};
        return internal::itransform<basic_vector<E, A>>(
            [](auto idx, E tval, E fval) { return mask[idx] ? tval : fval; },
            tval, fval);
    }

    template <auto V, typename LE, typename RE, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_mask<LE, A> tval, basic_mask<RE, A> fval) noexcept {
        static constexpr const_mask<simd_abi_traits<LE, A>::size, V> mask{};
        using E = common_size_type_t<LE, RE>;
        return internal::itransform<basic_mask<E, A>>(
            [](auto idx, bool tval, bool fval) {
                return mask[idx] ? tval : fval;
            },
            tval, fval);
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
                return select(internal::abi<A>, mask, tval, fval);
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
        unqualified_canonical_select<basic_mask<ME, MA>, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA, MA>> operator()(
        basic_mask<ME, MA> mask, basic_vector<E, LA> tval,
        basic_vector<E, RA> fval) noexcept {
        using A = common_abi_t<LA, RA, MA>;
        return select(internal::abi<A>, mask, tval, fval);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        common_size_with<typename M::value_type, typename R::value_type> &&
        common_size_with<typename M::value_type, typename L::value_type> &&
        unqualified_select<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L tval, R fval) noexcept {
        if constexpr (unqualified_extended_select<M, L, R>) {
            return select(mask, tval, fval);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> ME, simd_element_for<A> LE,
        simd_element_for<A> RE>
    requires common_size_with<ME, LE> && common_size_with<ME, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<common_size_type_t<LE, RE>, A> operator()(
        basic_mask<ME, A> mask, basic_mask<LE, A> tval,
        basic_mask<RE, A> fval) noexcept {
        if constexpr (unqualified_canonical_mask_select<basic_mask<ME, A>,
                          basic_mask<LE, A>, basic_mask<RE, A>, A>) {
            if consteval {
                return fallback(mask, tval, fval);
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return fallback(mask, tval, fval);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi LA, simd_abi RA,
        simd_element_for<LA> LE, simd_element_for<RA> RE>
    requires common_size_with<ME, LE> && common_size_with<ME, RE> &&
        (different_from<MA, LA> || different_from<MA, RA> ||
            different_from<LA, RA> || scalable_abi<MA> || scalable_abi<LA> ||
            scalable_abi<RA>) &&
        unqualified_canonical_mask_select<basic_mask<ME, MA>,
            basic_mask<LE, LA>, basic_mask<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<common_size_type_t<LE, RE>,
        common_abi_t<LA, RA, MA>> operator()(basic_mask<ME, MA> mask,
        basic_mask<LE, LA> tval, basic_mask<RE, RA> fval) noexcept {
        using A = common_abi_t<LA, RA, MA>;
        return select(internal::abi<A>, mask, tval, fval);
    }

    template <simd_mask M, simd_mask L, simd_mask R>
    requires (extended_mask<M> || extended_mask<L> || extended_mask<R>) &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        common_size_with<typename M::value_type, typename R::value_type> &&
        common_size_with<typename M::value_type, typename L::value_type> &&
        unqualified_mask_select<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L tval, R fval) noexcept {
        if constexpr (unqualified_extended_mask_select<M, L, R>) {
            return select(mask, tval, fval);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> ME, simd_element_for<A> E>
    requires common_size_with<ME, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        dx::zero_t tag, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bitdrop<basic_mask<ME, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return operator()(mask, dx::broadcast<E, A>(tag), val);
            } else {
                return select(internal::abi<A>, mask, tag, val);
            }
        } else {
            return operator()(mask, dx::broadcast<E, A>(tag), val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi RA,
        simd_element_for<RA> RE>
    requires common_size_with<ME, RE> &&
        (different_from<MA, RA> || scalable_abi<MA> || scalable_abi<RA>) &&
        (unqualified_canonical_bitdrop<basic_mask<ME, MA>,
             basic_vector<RE, RA>> ||
            regular_invocable<select_t, basic_mask<ME, MA>,
                basic_vector<RE, RA>, basic_vector<RE, RA>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<RE, common_abi_t<RA, MA>> operator()(
        basic_mask<ME, MA> mask, dx::zero_t tag,
        basic_vector<RE, RA> val) noexcept {
        using A = common_abi_t<RA, MA>;
        if constexpr (unqualified_canonical_bitdrop<basic_mask<ME, MA>,
                          basic_vector<RE, RA>>) {
            return select(internal::abi<A>, mask, tag, val);
        } else {
            return operator()(mask, dx::broadcast<RE, RA>(tag), val);
        }
    }

    template <simd_mask M, simd_vector R>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_bitdrop<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::zero_t tag, R fval) noexcept {
        if constexpr (unqualified_extended_bitdrop<M, R>) {
            return select(mask, tag, fval);
        } else {
            return operator()(
                dx::to_canonical(mask), tag, dx::to_canonical(fval));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> ME, simd_element_for<A> E>
    requires common_size_with<ME, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> val, dx::zero_t tag) noexcept {
        if constexpr (unqualified_canonical_bitkeep<basic_mask<ME, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<E, A>(tag));
            } else {
                return select(internal::abi<A>, mask, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<E, A>(tag));
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi RA,
        simd_element_for<RA> RE>
    requires common_size_with<ME, RE> &&
        (different_from<MA, RA> || scalable_abi<MA> || scalable_abi<RA>) &&
        (unqualified_canonical_bitkeep<basic_mask<ME, MA>,
             basic_vector<RE, RA>> ||
            regular_invocable<select_t, basic_mask<ME, MA>,
                basic_vector<RE, RA>, basic_vector<RE, RA>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<RE, common_abi_t<RA, MA>> operator()(
        basic_mask<ME, MA> mask, basic_vector<RE, RA> val,
        dx::zero_t tag) noexcept {
        using A = common_abi_t<RA, MA>;
        if constexpr (unqualified_canonical_bitkeep<basic_mask<ME, MA>,
                          basic_vector<RE, RA>>) {
            return select(internal::abi<A>, mask, val, tag);
        } else {
            return operator()(mask, val, dx::broadcast<RE, RA>(tag));
        }
    }

    template <simd_mask M, simd_vector R>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_bitkeep<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, R val, dx::zero_t tag) noexcept {
        if constexpr (unqualified_extended_bitkeep<M, R>) {
            return select(mask, val, tag);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(val), tag);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> ME, simd_element_for<A> E>
    requires common_size_with<ME, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        dx::all_bits_t tag, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bitfill<basic_mask<ME, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return operator()(mask, dx::broadcast<E, A>(tag), val);
            } else {
                return select(internal::abi<A>, mask, tag, val);
            }
        } else {
            return operator()(mask, dx::broadcast<E, A>(tag), val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi RA,
        simd_element_for<RA> RE>
    requires common_size_with<ME, RE> &&
        (different_from<MA, RA> || scalable_abi<MA> || scalable_abi<RA>) &&
        (unqualified_canonical_bitfill<basic_mask<ME, MA>,
             basic_vector<RE, RA>> ||
            regular_invocable<select_t, basic_mask<ME, MA>,
                basic_vector<RE, RA>, basic_vector<RE, RA>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<RE, common_abi_t<RA, MA>> operator()(
        basic_mask<ME, MA> mask, dx::all_bits_t tag,
        basic_vector<RE, RA> val) noexcept {
        using A = common_abi_t<RA, MA>;
        if constexpr (unqualified_canonical_bitfill<basic_mask<ME, MA>,
                          basic_vector<RE, RA>>) {
            return select(internal::abi<A>, mask, tag, val);
        } else {
            return operator()(mask, dx::broadcast<RE, RA>(tag), val);
        }
    }

    template <simd_mask M, simd_vector R>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_bitfill<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, dx::all_bits_t tag, R val) noexcept {
        if constexpr (unqualified_extended_bitfill<M, R>) {
            return select(mask, tag, val);
        } else {
            return operator()(
                dx::to_canonical(mask), tag, dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> ME, simd_element_for<A> E>
    requires common_size_with<ME, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> val, dx::all_bits_t tag) noexcept {
        if constexpr (unqualified_canonical_bitspill<basic_mask<ME, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<E, A>(tag));
            } else {
                return select(internal::abi<A>, mask, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<E, A>(tag));
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi RA,
        simd_element_for<RA> RE>
    requires common_size_with<ME, RE> &&
        (different_from<MA, RA> || scalable_abi<MA> || scalable_abi<RA>) &&
        (unqualified_canonical_bitspill<basic_mask<ME, MA>,
             basic_vector<RE, RA>> ||
            regular_invocable<select_t, basic_mask<ME, MA>,
                basic_vector<RE, RA>, basic_vector<RE, RA>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<RE, common_abi_t<RA, MA>> operator()(
        basic_mask<ME, MA> mask, basic_vector<RE, RA> val,
        dx::all_bits_t tag) noexcept {
        using A = common_abi_t<RA, MA>;
        if constexpr (unqualified_canonical_bitspill<basic_mask<ME, MA>,
                          basic_vector<RE, RA>>) {
            return select(internal::abi<A>, mask, val, tag);
        } else {
            return operator()(mask, val, dx::broadcast<RE, RA>(tag));
        }
    }

    template <simd_mask M, simd_vector R>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_bitspill<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, R val, dx::all_bits_t tag) noexcept {
        if constexpr (unqualified_extended_bitspill<M, R>) {
            return select(mask, val, tag);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(val), tag);
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        constexpr auto V =
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask)();
        if constexpr (unqualified_canonical_selecti<M, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallbacki<V>(lhs, rhs);
            } else {
                return select<V>(internal::abi<A>, lhs, rhs);
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
        unqualified_canonical_selecti<M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        using L = basic_vector<E, LA>;
        using R = basic_vector<E, RA>;
        constexpr auto V = internal::select_mask<M, L, R>()();
        return select<V>(internal::abi<A>, mask, lhs, rhs);
    }

    template <const_mask_like M, fixed_width_vector L, fixed_width_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        unqualified_selecti<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_selecti<M, L, R>) {
            constexpr auto V = internal::select_mask<M, L, R>()();
            return select<V>(lhs, rhs);
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, basic_mask<E, A> lhs, basic_mask<E, A> rhs) noexcept {
        constexpr auto V =
            dx::to_compatible_const_mask<basic_mask<E, A>>(mask)();
        if constexpr (unqualified_canonical_mask_selecti<M, basic_mask<E, A>,
                          basic_mask<E, A>>) {
            if consteval {
                return fallbacki<V>(lhs, rhs);
            } else {
                return select<V>(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallbacki<V>(lhs, rhs);
        }
    }

    template <typename M, fixed_width_abi LA, fixed_width_abi RA,
        simd_element_for<LA> E>
    requires simd_element_for<E, RA> && different_from<LA, RA> &&
        const_mask_for<M, basic_mask<E, LA>> &&
        const_mask_for<M, basic_mask<E, RA>> &&
        unqualified_canonical_selecti<M, basic_mask<E, LA>, basic_mask<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_mask<E, LA> lhs, basic_mask<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        using L = basic_mask<E, LA>;
        using R = basic_mask<E, RA>;
        constexpr auto V = internal::select_mask<M, L, R>()();
        return select<V>(internal::abi<A>, mask, lhs, rhs);
    }

    template <typename M, fixed_width_mask L, fixed_width_mask R>
    requires (extended_mask<L> || extended_mask<R>) && const_mask_for<M, L> &&
        const_mask_for<M, R> && unqualified_selecti<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_selecti<M, L, R>) {
            constexpr auto V = internal::select_mask<M, L, R>()();
            return select<V>(lhs, rhs);
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, dx::zero_t tag, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bitdropi<M, basic_vector<E, A>>) {
            if consteval {
                return operator()(mask, dx::broadcast<E, A>(tag), val);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, M>;
                return select<V>(internal::abi<A>, tag, val);
            }
        } else {
            return operator()(mask, dx::broadcast<E, A>(tag), val);
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, dx::zero_t tag, basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bitdropi<M, basic_mask<E, A>>) {
            if consteval {
                return operator()(mask, dx::broadcast<E, A>(false_type{}), val);
            } else {
                constexpr auto V = const_mask_v<basic_mask<E, A>, M>;
                return select<V>(internal::abi<A>, tag, val);
            }
        } else {
            return operator()(mask, dx::broadcast<E, A>(false_type{}), val);
        }
    }

    template <typename M, extended_class R>
    requires fixed_width_class<R> && const_mask_for<M, R> &&
        unqualified_bitdropi<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::zero_t tag, R fval) noexcept {
        if constexpr (unqualified_extended_bitdropi<M, R>) {
            constexpr auto V = const_mask_v<R, M>;
            return select<V>(tag, fval);
        } else {
            return operator()(mask, tag, dx::to_canonical(fval));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, dx::zero_t tag) noexcept {
        if constexpr (unqualified_canonical_bitkeepi<M, basic_vector<E, A>>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<E, A>(tag));
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, M>;
                return select<V>(internal::abi<A>, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<E, A>(tag));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, basic_mask<E, A> val, dx::zero_t tag) noexcept {
        if constexpr (unqualified_canonical_bitkeepi<M, basic_mask<E, A>>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<E, A>(false_type{}));
            } else {
                constexpr auto V = const_mask_v<basic_mask<E, A>, M>;
                return select<V>(internal::abi<A>, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<E, A>(false_type{}));
        }
    }

    template <typename M, extended_class R>
    requires fixed_width_class<R> && const_mask_for<M, R> &&
        unqualified_bitkeepi<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, R val, dx::zero_t tag) noexcept {
        if constexpr (unqualified_extended_bitkeepi<M, R>) {
            constexpr auto V = const_mask_v<R, M>;
            return select<V>(val, tag);
        } else {
            return operator()(mask, dx::to_canonical(val), tag);
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, dx::all_bits_t tag, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bitfilli<M, basic_vector<E, A>>) {
            if consteval {
                return operator()(mask, dx::broadcast<E, A>(tag), val);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, M>;
                return select<V>(internal::abi<A>, tag, val);
            }
        } else {
            return operator()(mask, dx::broadcast<E, A>(tag), val);
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, dx::all_bits_t tag, basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bitfilli<M, basic_mask<E, A>>) {
            if consteval {
                return operator()(mask, dx::broadcast<E, A>(true_type{}), val);
            } else {
                constexpr auto V = const_mask_v<basic_mask<E, A>, M>;
                return select<V>(internal::abi<A>, tag, val);
            }
        } else {
            return operator()(mask, dx::broadcast<E, A>(true_type{}), val);
        }
    }

    template <typename M, extended_class R>
    requires fixed_width_class<R> && const_mask_for<M, R> &&
        unqualified_bitfilli<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, dx::all_bits_t tag, R val) noexcept {
        if constexpr (unqualified_extended_bitfilli<M, R>) {
            constexpr auto V = const_mask_v<R, M>;
            return select<V>(tag, val);
        } else {
            return operator()(mask, tag, dx::to_canonical(val));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, dx::all_bits_t tag) noexcept {
        if constexpr (unqualified_canonical_bitspilli<M, basic_vector<E, A>>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<E, A>(tag));
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, M>;
                return select<V>(internal::abi<A>, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<E, A>(tag));
        }
    }

    template <typename M, fixed_width_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, basic_mask<E, A> val, dx::all_bits_t tag) noexcept {
        if constexpr (unqualified_canonical_bitspilli<M, basic_mask<E, A>>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<E, A>(true_type{}));
            } else {
                constexpr auto V = const_mask_v<basic_mask<E, A>, M>;
                return select<V>(internal::abi<A>, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<E, A>(true_type{}));
        }
    }

    template <typename M, extended_class R>
    requires fixed_width_class<R> && const_mask_for<M, R> &&
        unqualified_bitspilli<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, R val, dx::all_bits_t tag) noexcept {
        if constexpr (unqualified_extended_bitspilli<M, R>) {
            constexpr auto V = const_mask_v<R, M>;
            return select<V>(val, tag);
        } else {
            return operator()(mask, dx::to_canonical(val), tag);
        }
    }

    template <simd_mask M, typename L, typename R>
    requires (simd_class<L> && !simd_class<R> && broadcastable_to<R, L>) ||
        (simd_class<R> && !simd_class<L> && broadcastable_to<L, R>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(M mask, L lhs, R rhs) noexcept
    requires requires {
        select_t::operator()(mask, select_t::selective_cast<L, R>(lhs),
            select_t::selective_cast<L, R>(rhs));
    }
    {
        using A =
            common_abi_t<typename M::abi_type, decltype(selective_abi<L, R>())>;
        if constexpr (canonical_mask<M> &&
            (canonical_class<L> || canonical_class<R>)) {
            if constexpr (requires {
                              select(internal::abi<A>, mask, lhs, rhs);
                          }) {
                if consteval {
                    return operator()(mask, select_t::selective_cast<L, R>(lhs),
                        select_t::selective_cast<L, R>(rhs));
                } else {
                    return select(internal::abi<A>, mask, lhs, rhs);
                }
            } else {
                return operator()(mask, select_t::selective_cast<L, R>(lhs),
                    select_t::selective_cast<L, R>(rhs));
            }
        } else if constexpr (simd_class<L> || simd_class<R>) {
            if constexpr (requires { select(mask, lhs, rhs); }) {
                return select(mask, lhs, rhs);
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
        select_t::operator()(mask, select_t::selective_cast<L, R>(lhs),
            select_t::selective_cast<L, R>(rhs));
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
            if constexpr (requires { select<V>(internal::abi<A>, lhs, rhs); }) {
                if consteval {
                    return operator()(mask, select_t::selective_cast<L, R>(lhs),
                        select_t::selective_cast<L, R>(rhs));
                } else {
                    return select<V>(internal::abi<A>, lhs, rhs);
                }
            } else {
                return operator()(mask, select_t::selective_cast<L, R>(lhs),
                    select_t::selective_cast<L, R>(rhs));
            }
        } else if constexpr (requires { select<V>(lhs, rhs); }) {
            return select<V>(lhs, rhs);
        } else if constexpr (simd_class<L>) {
            return operator()(mask, dx::to_canonical(lhs), rhs);
        } else {
            return operator()(mask, lhs, dx::to_canonical(rhs));
        }
    }
};

template <auto V>
struct selecti_t {};
template <integral auto V>
struct selecti_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_const_mask_t<T, V>;

public:
    template <simd_class L, simd_class R>
    requires requires {
        typename mask_type<L>;
        typename mask_type<R>;
        requires regular_invocable<select_t, mask_type<R>, L, R>;
        requires regular_invocable<select_t, mask_type<L>, L, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (mask_type<L>::width > mask_type<R>::width) {
            constexpr mask_type<L> mask{};
            return select_t::operator()(mask, lhs, rhs);
        } else {
            constexpr mask_type<R> mask{};
            return select_t::operator()(mask, lhs, rhs);
        }
    }

    template <simd_class L, typename R>
    requires (!simd_class<R>) && requires { typename mask_type<L>; } &&
        regular_invocable<select_t, mask_type<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<L> mask{};
        return select_t::operator()(mask, lhs, rhs);
    }

    template <typename L, simd_class R>
    requires (!simd_class<L>) && requires { typename mask_type<R>; } &&
        regular_invocable<select_t, mask_type<R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<R> mask{};
        return select_t::operator()(mask, lhs, rhs);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <integral auto V>
inline constexpr internal::selecti_t<V> selecti{};
DPL_EXPORT inline constexpr internal::select_t select{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
