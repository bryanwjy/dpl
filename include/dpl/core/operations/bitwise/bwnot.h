// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwandnot.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/all_bits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwnot(...) noexcept = delete;

struct DPL_EMPTY_BASES bwnot_t :
    private bitwise_base<bwnot_t>,
    private maskable_transform_base<bwnot_t> {
    using bitwise_base<bwnot_t>::operator();
    using maskable_transform_base<bwnot_t>::operator();
};

template <>
struct operation_signature<bwnot_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
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
            return bwandnot_t::operator()(val, dx::all_bits);
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
        cpo_invocable<pack_mask_t, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_mask<E, A> val) noexcept {
        if constexpr (cpo_invocable<bwandnot_t, basic_mask<E, A>, true_type>) {
            return bwandnot_t::operator()(val, true_type{});
        } else {
            return dx::initialize<E, A>(~dx::pack_mask(val));
        }
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mbwnot =
    (!simd_type<S> || same_as<common_abi_t<S, T>, simd_abi_type_t<S>>) &&
    requires(S src, M mask, T val) {
        bwnot(internal::abi<conditional_t<simd_type<S>, S, T>>, src, mask, val);
    };

template <>
struct canonical_impl<bwnot_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

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

    template <canonical_vector S, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwnot<S, mask_t<S>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, mask_t<S> mask, T val) noexcept {
        return bwnot(internal::abi<S>, src, mask, val);
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwnot<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, cmask_t<S, M> cmask, T val) noexcept {
        return bwnot(internal::abi<S>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwnot<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return bwnot(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires unqualified_canonical_mbwnot<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return bwnot(internal::abi<T>, zero, cmask, val);
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
concept unqualified_extended_mbwnot =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<bwnot_t, T>>) &&
    requires {
        {
            bwnot(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<
            conditional_t<simd_type<S>, S, cpo_result_t<bwnot_t, T>>>;
    };
template <>
struct extended_impl<bwnot_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

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

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwnot<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return bwnot( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwnot<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return bwnot(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mbwnot<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return bwnot(zero, cmask, __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwnot_t bwnot{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
