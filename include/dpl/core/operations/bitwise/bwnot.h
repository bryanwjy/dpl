// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwandnot.h"
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
    using bitset_t DPL_NODEBUG = bitset<sizeof(E) * char_bit_v>;

    template <simd_abi A, simd_element_for<A> E>
    requires cpo_invocable<bwandnot_t, basic_mask<E, A>, all_bits_t> ||
        (sizeof(bitset_t<E>) == sizeof(E))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        if constexpr (cpo_invocable<bwandnot_t, basic_mask<E, A>, all_bits_t>) {
            return bwandnot_t::operator()(dx::all_bits, val);
        } else {
            using bit_type = bit_type_t<sizeof(E) * char_bit_v>;
            return internal::transform<basic_vector<E, A>>(
                [](auto val) {
                    auto const negated = ~__DPL bit_cast<bitset_t<E>>(val);
                    return __DPL bit_cast<E>(negated);
                },
                val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires cpo_invocable<bwandnot_t, basic_mask<E, A>, true_type> ||
        cpo_invocable<to_bitset_t, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_mask<E, A> val) noexcept {
        if constexpr (cpo_invocable<bwandnot_t, basic_mask<E, A>, true_type>) {
            return bwandnot_t::operator()(true_type{}, val);
        } else {
            return dx::from_bitset<E, A>(~dx::to_bitset(val));
        }
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mbwnot = cpo_invocable<bwnot_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<bwnot_t, T>>) && requires {
        {
            bwnot(internal::abi<cpo_result_t<bwnot_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<bwnot_t, T>>;
    };

template <>
struct canonical_impl<bwnot_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(basic_mask<E, A> val) noexcept
    requires requires { bwnot(internal::abi<A>, val); }
    {
        return bwnot(internal::abi<A>, val);
    }

    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { bwnot(internal::abi<A>, val); }
    {
        return bwnot(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwnot<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return bwnot(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mbwnot<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return bwnot(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwnot<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return bwnot(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mbwnot<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return bwnot(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_mask_bwnot = requires {
    { bwnot(internal::declarg<T>()) } -> mask_with_common_abi<A>;
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_bwnot = requires {
    { bwnot(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mbwnot = cpo_invocable<bwnot_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<bwnot_t, T>>) &&
    requires {
        {
            bwnot(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<bwnot_t, T>>;
    };
template <>
struct extended_impl<bwnot_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <extended_mask T>
    requires unqualified_extended_mask_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return bwnot(__DPL forward<T>(val));
    }

    template <extended_vector T>
    requires unqualified_extended_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return bwnot(__DPL forward<T>(val));
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwnot<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return bwnot( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwnot<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return bwnot( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwnot<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return bwnot(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<bwnot_t, T> M>
    requires unqualified_extended_mbwnot<dx::zero_t,
        launder_cmask_t<cpo_result_t<bwnot_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return bwnot(zero, dx::to_const_mask<cpo_result_t<bwnot_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::bwnot_t bwnot{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
