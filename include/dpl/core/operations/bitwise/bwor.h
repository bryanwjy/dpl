// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/internal/broadcasting.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwor(...) noexcept = delete;

struct DPL_EMPTY_BASES bwor_t :
    public bitwise_base<bwor_t>,
    public maskable_transform_base<bwor_t> {
    using bitwise_base<bwor_t>::operator();
    using maskable_transform_base<bwor_t>::operator();
};

template <>
struct operation_signature<bwor_t> {
    template <typename L, typename R>
    requires simd_type<L> || simd_type<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<bwor_t> : binary_canonical_broadcaster<bwor_t> {
    template <canonical_vector T>
    requires requires { typename bit_representation_t<simd_element_type_t<T>>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T lhs, T rhs) noexcept {
        return internal::transform<T>(
            [](auto lhs, auto rhs) {
                auto bits = __DPL to_bit_representation(lhs) |
                    __DPL to_bit_representation(rhs);
                return __DPL bit_cast<simd_element_type_t<T>>(
                    bits.reinitialize());
            },
            lhs, rhs);
    }

    template <canonical_mask L, equivalent_mask_with<L> R>
    requires canonical_mask<R> && cpo_invocable<to_bitset_t, L> &&
        cpo_invocable<to_bitset_t, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr common_canonical_simd_t<L, R>
        DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using T = common_canonical_simd_t<L, R>;
        return dx::from_bitset<T>(dx::to_bitset(lhs) | dx::to_bitset(rhs));
    }

    using binary_canonical_broadcaster<bwor_t>::operator();
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_bwor = requires {
    {
        bwor(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename T>
concept unqualified_canonical_mbwor_base =
    cpo_invocable<bwor_t, L, R> && requires {
        {
            bwor(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<T>;
    };

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_mbwor =
    unqualified_canonical_mbwor_base<T, M, L, R, T>;

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_zmbwor =
    unqualified_canonical_mbwor_base<dx::zero_t, M, L, R, T>;

template <>
struct canonical_impl<bwor_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<bwor_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_bwor<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        return bwor(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_mask L, common_mask_with<L> R>
    requires canonical_mask<R> && unqualified_canonical_bwor<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        return bwor(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_bwor<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return bwor(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_bwor<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return bwor(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_type R>
    requires unqualified_canonical_mbwor<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return bwor(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_mbwor<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return bwor(internal::abi<T>, src, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R>
    requires unqualified_canonical_zmbwor<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return bwor(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_zmbwor<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return bwor(internal::abi<T>, zero, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename L, typename R>
concept unqualified_extended_mask_bwor = requires {
    {
        bwor(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<common_canonical_simd_t<L, R>>;
};

template <typename L, typename R, typename T>
concept unqualified_extended_bwor = requires {
    {
        bwor(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mbwor_base =
    cpo_invocable<bwor_t, L, R> && requires {
        {
            bwor(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<bwor_t, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mbwor =
    equivalent_vector_with<S, cpo_result_t<bwor_t, L, R>> &&
    unqualified_extended_mbwor_base<S, M, L, R>;

template <typename M, typename L, typename R>
concept unqualified_extended_zmbwor =
    unqualified_extended_mbwor_base<dx::zero_t, M, L, R>;

template <>
struct extended_impl<bwor_t> {
public:
    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_mask_bwor<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwor(__DPL forward<L>(lhs), __DPL forward<L>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_bwor<L, R, common_canonical_simd_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwor(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_bwor<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwor(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_bwor<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwor(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mbwor<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return bwor(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwor<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return bwor(src, dx::to_const_mask<S>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_mask_for<bwor_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmbwor<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return bwor(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<bwor_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmbwor<
            launder_cmask_t<cpo_result_t<bwor_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return bwor(zero, dx::to_const_mask<cpo_result_t<bwor_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::bwor_t bwor{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
