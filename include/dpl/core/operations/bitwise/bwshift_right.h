// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/vshift_vector_for.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwshift_right(...) noexcept = delete;

struct DPL_EMPTY_BASES bwshift_right_t :
    public bitwise_base<bwshift_right_t>,
    public maskable_transform_base<bwshift_right_t> {
    using bitwise_base<bwshift_right_t>::operator();
    using maskable_transform_base<bwshift_right_t>::operator();
};

template <>
struct operation_signature<bwshift_right_t> {
    template <simd_type L, integral_constant_like R>
    requires (!simd_vector<R>)
    static consteval void operator()(L&&, R) noexcept {}
    template <simd_type L>
    static consteval void operator()(L&&, size_t) noexcept {}
    template <simd_vector L, simd_vector R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<bwshift_right_t> {
    template <canonical_vector T>
    requires requires { typename bit_representation_t<simd_element_type_t<T>>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T lhs, unsigned_canonical_vector_t<T> rhs) noexcept {
        return internal::transform<T>(
            [](auto lhs, auto rhs) {
                using E DPL_NODEBUG = simd_element_type_t<T>;
                if constexpr (signed_integral<E>) {
                    return static_cast<E>(lhs >> rhs);
                } else {
                    auto bits = __DPL to_bit_representation(lhs) >> rhs;
                    return __DPL bit_cast<simd_element_type_t<T>>(
                        bits.reinitialize());
                }
            },
            lhs, rhs);
    }

    template <canonical_vector T>
    requires requires { typename bit_representation_t<simd_element_type_t<T>>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T lhs, size_t rhs) noexcept {
        return internal::transform<T>(
            [rhs](auto lhs) {
                using E DPL_NODEBUG = simd_element_type_t<T>;
                if constexpr (signed_integral<E>) {
                    return static_cast<E>(lhs >> rhs);
                } else {
                    auto bits = __DPL to_bit_representation(lhs) >> rhs;
                    return __DPL bit_cast<simd_element_type_t<T>>(
                        bits.reinitialize());
                }
            },
            lhs);
    }

    template <canonical_mask T>
    requires cpo_invocable<to_bitset_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T lhs, size_t rhs) noexcept {
        return dx::from_bitset<T>(dx::to_bitset(lhs) >> rhs);
    }
};

template <typename T, typename N = size_t>
concept unqualified_canonical_bwshift_right = requires {
    {
        bwshift_right(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<N>())
    } -> same_as<T>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mbwshift_right_base =
    cpo_invocable<bwshift_right_t, T, N> && requires {
        {
            bwshift_right(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<T>(),
                internal::declarg<N>())
        } -> same_as<T>;
    };

template <typename M, typename T, typename N = size_t>
concept unqualified_canonical_mbwshift_right =
    unqualified_canonical_mbwshift_right_base<T, M, T, N>;

template <typename M, typename T, typename N = size_t>
concept unqualified_canonical_zmbwshift_right =
    unqualified_canonical_mbwshift_right_base<dx::zero_t, M, T, N>;

template <>
struct canonical_impl<bwshift_right_t> {
private:
    template <typename T>
    using vcount_t DPL_NODEBUG = unsigned_canonical_vector_t<T>;

public:
    template <canonical_mask T>
    requires unqualified_canonical_bwshift_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return bwshift_right(internal::abi<T>, val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_bwshift_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return bwshift_right(internal::abi<T>, val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwshift_right<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, size_t count) noexcept {
        return bwshift_right(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mbwshift_right<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T val, size_t count) noexcept {
        return bwshift_right(
            internal::abi<T>, src, dx::to_const_mask<T>(mask), val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmbwshift_right<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, size_t count) noexcept {
        return bwshift_right(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmbwshift_right<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M mask, T val, size_t count) noexcept {
        return bwshift_right(
            internal::abi<T>, zero, dx::to_const_mask<T>(mask), val, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N count) noexcept {
        return bwshift_right(internal::abi<T>, val, count);
    }

    template <canonical_mask T, integral_constant_like N>
    requires unqualified_canonical_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N count) noexcept {
        return bwshift_right(internal::abi<T>, val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mbwshift_right<simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, N count) noexcept {
        return bwshift_right(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mbwshift_right<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T val, N count) noexcept {
        return bwshift_right(
            internal::abi<T>, src, dx::to_const_mask<T>(mask), val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_zmbwshift_right<simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, N count) noexcept {
        return bwshift_right(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_zmbwshift_right<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M mask, T val, N count) noexcept {
        return bwshift_right(
            internal::abi<T>, zero, dx::to_const_mask<T>(mask), val, count);
    }

    ///
    template <canonical_vector T>
    requires unqualified_canonical_bwshift_right<T, vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T lhs, vcount_t<T> rhs) noexcept {
        return bwshift_right(internal::abi<T>, lhs, rhs);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwshift_right<simd_mask_type_t<T>, T,
        vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T lhs, vcount_t<T> rhs) noexcept {
        return bwshift_right(internal::abi<T>, src, mask, lhs, rhs);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mbwshift_right<launder_cmask_t<T, M>, T,
        vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T lhs, vcount_t<T> rhs) noexcept {
        return bwshift_right(
            internal::abi<T>, src, dx::to_const_mask<T>(mask), lhs, rhs);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmbwshift_right<simd_mask_type_t<T>, T,
        vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T lhs, vcount_t<T> rhs) noexcept {
        return bwshift_right(internal::abi<T>, zero, mask, lhs, rhs);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmbwshift_right<launder_cmask_t<T, M>, T,
        vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M mask, T lhs, vcount_t<T> rhs) noexcept {
        return bwshift_right(
            internal::abi<T>, zero, dx::to_const_mask<T>(mask), lhs, rhs);
    }
};

template <typename T, typename N = size_t>
concept unqualified_extended_bwshift_right = requires {
    {
        bwshift_right(internal::declarg<T>(), internal::declarg<N>())
    } -> equivalent_vector_with<T>;
};

template <typename T, typename N = size_t>
concept unqualified_extended_mask_bwshift_right = requires {
    {
        bwshift_right(internal::declarg<T>(), internal::declarg<N>())
    } -> equivalent_mask_with<T>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mbwshift_right_base =
    cpo_invocable<bwshift_right_t, T, N> && requires {
        {
            bwshift_right(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<T>;
    };

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mbwshift_right =
    unqualified_extended_mbwshift_right_base<S, M, T, N> &&
    equivalent_vector_with<S, T>;

template <typename M, typename T, typename N = size_t>
concept unqualified_extended_zmbwshift_right =
    unqualified_extended_mbwshift_right_base<dx::zero_t, M, T, N>;

template <>
struct extended_impl<bwshift_right_t> {
public:
    template <extended_mask T>
    requires unqualified_extended_bwshift_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& lhs, size_t shift) {
        return bwshift_right(__DPL forward<T>(lhs), shift);
    }

    template <extended_mask T, integral_constant_like N>
    requires unqualified_extended_mask_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& lhs, N shift) {
        return bwshift_right(__DPL forward<T>(lhs), shift);
    }
    ///

    template <simd_vector T, vshift_vector_for<T> N>
    requires (extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& lhs, N&& rhs) {
        return bwshift_right(__DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        vshift_vector_for<T> N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T> ||
                 extended_vector<N>) &&
        unqualified_extended_mbwshift_right<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& lhs, N&& rhs) {
        return bwshift_right(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        vshift_vector_for<T> N>
    requires (extended_vector<S> || extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_mbwshift_right<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& lhs, N&& rhs) {
        return bwshift_right(src, dx::to_const_mask<S>(mask),
            __DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }

    template <simd_vector T, vshift_vector_for<T> N, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_zmbwshift_right<M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& lhs, N&& rhs) {
        return bwshift_right(zero, __DPL forward<M>(mask),
            __DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }

    template <simd_vector T, vshift_vector_for<T> N, const_mask_for<T> M>
    requires (extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_zmbwshift_right<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, T&& lhs, N&& rhs) {
        return bwshift_right(zero, dx::to_const_mask<T>(mask),
            __DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }
    ///
    template <extended_vector T>
    requires unqualified_extended_bwshift_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t count) {
        return bwshift_right(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, size_t count) {
        return bwshift_right( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, size_t count) {
        return bwshift_right( __DPL forward<S>(src),
            dx::to_const_mask<S>(mask), __DPL forward<T>(val), count);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmbwshift_right<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, size_t count) {
        return bwshift_right(zero, mask, __DPL forward<T>(val), count);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zmbwshift_right<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t count) {
        return bwshift_right(
            zero, dx::to_const_mask<T>(cmask), __DPL forward<T>(val), count);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N count) {
        return bwshift_right(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, N count) {
        return bwshift_right( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N count) {
        return bwshift_right( __DPL forward<S>(src),
            dx::to_const_mask<S>(cmask), __DPL forward<T>(val), count);
    }

    template <simd_vector T, integral_constant_like N, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmbwshift_right<M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N count) {
        return bwshift_right(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), count);
    }

    template <extended_vector T, integral_constant_like N, const_mask_for<T> M>
    requires unqualified_extended_zmbwshift_right<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, N count) {
        return bwshift_right(
            zero, dx::to_const_mask<T>(cmask), __DPL forward<T>(val), count);
    }
};

template <size_t N>
struct bwshift_righti_t {
    template <typename... Ts>
    requires cpo_invocable<bwshift_right_t, Ts..., size_constant<N>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... val) noexcept(
        (... && (!simd_type<Ts> || canonical_simd_type<Ts>))) {
        constexpr size_constant<N> shift;
        return bwshift_right_t::operator()(__DPL forward<Ts>(val)..., shift);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::bwshift_right_t bwshift_right{};
template <size_t N>
inline constexpr internal::bwshift_righti_t<N> bwshift_righti{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
