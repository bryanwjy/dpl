// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/byteswap.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void byteswap(...) noexcept = delete;

struct byteswap_t :
    public bit_manipulation_base<byteswap_t>,
    public maskable_transform_base<byteswap_t> {
    using operation_base<byteswap_t>::operator();
    using maskable_transform_base<byteswap_t>::operator();
};

template <>
struct operation_signature<byteswap_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <>
struct fallback_impl<byteswap_t> {
    template <fixed_width_abi A, simd_element_for<A> E>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        using ubit = unsigned_representation_t<E>;
        return internal::transform<basic_vector<ubit, A>>(
            [](auto val) {
                auto const count = __DPL byteswap(__DPL bit_cast<ubit>(val));
                return static_cast<ubit>(count);
            },
            val);
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mbyteswap = cpo_invocable<byteswap_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<byteswap_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            byteswap(internal::abi<cpo_result_t<byteswap_t, T>>, src, mask, val)
        } -> same_as<cpo_result_t<byteswap_t, T>>;
    };

template <>
struct canonical_impl<byteswap_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires integral<E> && requires { byteswap(internal::abi<A>, val); }
    {
        return byteswap(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires integral<simd_element_type_t<T>> &&
        unqualified_canonical_mbyteswap<T, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return byteswap(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires integral<simd_element_type_t<T>> &&
        unqualified_canonical_mbyteswap<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return byteswap(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbyteswap<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return byteswap(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mbyteswap<dx::zero_t, launder_cmask_t<T, M>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return byteswap(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_byteswap = requires {
    { byteswap(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mbyteswap = cpo_invocable<byteswap_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<byteswap_t, T>>) &&
    requires {
        {
            byteswap(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<
            conditional_t<simd_type<S>, S, cpo_result_t<byteswap_t, T>>>;
    };

template <>
struct extended_impl<byteswap_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_byteswap<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return byteswap(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbyteswap<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return byteswap( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, const_mask_for<S> M,
        common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbyteswap<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return byteswap( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<byteswap_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbyteswap<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return byteswap(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, result_cmask_for<byteswap_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mbyteswap<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return byteswap(
            zero, dx::to_const_mask<T>(cmask), __DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::byteswap_t byteswap{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
