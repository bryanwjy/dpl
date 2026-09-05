// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwandnot.h"
#include "dpl/core/operations/internal/broadcasting.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/all_bits.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwnot(...) noexcept = delete;

struct DPL_EMPTY_BASES bwnot_t :
    public bitwise_base<bwnot_t>,
    public maskable_transform_base<bwnot_t> {
    using bitwise_base<bwnot_t>::operator();
    using maskable_transform_base<bwnot_t>::operator();
};

template <>
struct operation_signature<bwnot_t> {
    static consteval void operator()(simd_type auto&&) noexcept {}
};

template <>
struct fallback_impl<bwnot_t> {
    template <typename E>
    using bitset_t DPL_NODEBUG = bitset<__DPL type_bit_v<E>>;

    template <canonical_vector T>
    requires cpo_invocable<bwandnot_t, all_bits_t, T> ||
        requires { typename bit_representation_t<simd_element_type_t<T>>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        if constexpr (cpo_invocable<bwandnot_t, all_bits_t, T>) {
            return bwandnot_t::operator()(dx::all_bits, val);
        } else {
            return internal::transform<T>(
                [](auto val) {
                    return __DPL bit_cast<simd_element_type_t<T>>(
                        ~__DPL to_bit_representation(val));
                },
                val);
        }
    }

    template <canonical_mask T>
    requires cpo_invocable<to_bitset_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        return dx::from_bitset<T>(~dx::to_bitset(val));
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mbwnot_base =
    cpo_invocable<bwnot_t, T> && requires {
        {
            bwnot(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<T>())
        } -> same_as<T>;
    };

template <typename M, typename T>
concept unqualified_canonical_mbwnot =
    unqualified_canonical_mbwnot_base<T, M, T>;

template <typename M, typename T>
concept unqualified_canonical_zmbwnot =
    unqualified_canonical_mbwnot_base<dx::zero_t, M, T>;

template <>
struct canonical_impl<bwnot_t> {
public:
    template <canonical_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { bwnot(internal::abi<T>, val); }
    {
        return bwnot(internal::abi<T>, val);
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { bwnot(internal::abi<T>, val); }
    {
        return bwnot(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwnot<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return bwnot(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mbwnot<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        return bwnot(internal::abi<T>, src, dx::to_const_mask<T>(mask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmbwnot<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return bwnot(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmbwnot<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M mask, T val) noexcept {
        return bwnot(internal::abi<T>, zero, dx::to_const_mask<T>(mask), val);
    }
};

template <typename T>
concept unqualified_extended_bwnot = requires {
    { bwnot(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mbwnot_base =
    cpo_invocable<bwnot_t, T> && requires {
        {
            bwnot(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<T>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mbwnot =
    equivalent_vector_with<T, S> && unqualified_extended_mbwnot_base<S, M, T>;

template <typename M, typename T>
concept unqualified_extended_zmbwnot =
    unqualified_extended_mbwnot_base<dx::zero_t, M, T>;

template <>
struct extended_impl<bwnot_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return bwnot(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwnot<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return bwnot( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwnot<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val) {
        return bwnot( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmbwnot<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return bwnot(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zmbwnot<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, T&& val) {
        return bwnot(zero, dx::to_const_mask<T>(mask), __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::bwnot_t bwnot{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
