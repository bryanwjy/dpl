// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/byteswap.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void byteswap(...) noexcept = delete;

struct byteswap_t :
    private bit_manipulation_base<byteswap_t>,
    private maskable_transform_base<byteswap_t> {
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
concept unqualified_canonical_mbyteswap =
    (!simd_type<S> || same_as<common_abi_t<S, T>, simd_abi_type_t<S>>) &&
    requires(S src, M mask, T val) {
        byteswap(
            internal::abi<conditional_t<simd_type<S>, S, T>>, src, mask, val);
    };

template <>
struct canonical_impl<byteswap_t> {
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
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { byteswap(internal::abi<A>, val); }
    {
        return byteswap(internal::abi<A>, val);
    }

    template <canonical_vector S, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mbyteswap<S, mask_t<S>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, mask_t<S> mask, T val) noexcept {
        return byteswap(internal::abi<S>, src, mask, val);
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mbyteswap<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, cmask_t<S, M> cmask, T val) noexcept {
        return byteswap(internal::abi<S>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbyteswap<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return byteswap(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires unqualified_canonical_mbyteswap<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return byteswap(internal::abi<T>, zero, cmask, val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_byteswap = requires {
    { byteswap(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mbyteswap =
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
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <extended_vector T>
    requires unqualified_extended_byteswap<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return byteswap(__DPL forward<T>(val));
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbyteswap<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return byteswap( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbyteswap<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return byteswap( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbyteswap<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return byteswap(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mbyteswap<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return byteswap(zero, cmask, __DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::byteswap_t byteswap{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
