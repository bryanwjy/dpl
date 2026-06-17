// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/infinity.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isfinite(...) noexcept = delete;

struct DPL_EMPTY_BASES isfinite_t :
    private math_operation_base<isfinite_t>,
    private maskable_predicate_base<isfinite_t> {
    using math_operation_base<isfinite_t>::operator();
    using maskable_predicate_base<isfinite_t>::operator();
};

template <>
struct operation_signature<isfinite_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isfinite_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

    template <typename T>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        auto const inf = dx::broadcast<A, E>(dx::infinity);
        return dx::cmpneq(dx::bwand(val, inf), inf);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_mask<E, A> mask, basic_vector<E, A> val) noexcept {
        auto const inf = dx::broadcast<A, E>(dx::infinity);
        return dx::cmpneq(mask, dx::bwand(val, inf), inf);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> && floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(cmask_t<T, M> cmask, T val) noexcept {
        auto const inf = dx::broadcast<T>(dx::infinity);
        return dx::cmpneq(cmask, dx::bwand(val, inf), inf);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isfinite = requires {
    {
        isfinite(internal::abi<A>, internal::declarg<T>())
    } -> same_as<basic_mask<simd_element_type_t<T>, A>>;
};

template <typename S, typename T>
concept unqualified_canonical_misfinite = canonical_mask<S> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isfinite_t, T>>) && requires {
        {
            isfinite(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isfinite_t, T>>;
    };

template <>
struct canonical_impl<isfinite_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpeq_t, T>;

    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_isfinite<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T arg) noexcept {
        return isfinite(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misfinite<mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(mask_t<T> src, T val) noexcept {
        return isfinite(internal::abi<T>, src, val);
    }

    template <canonical_vector T, imask_t<T> M>
    requires unqualified_canonical_misfinite<cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(cmask_t<T, M> cmask, T val) noexcept {
        return isfinite(internal::abi<T>, cmask, val);
    }
};

template <typename T>
concept unqualified_extended_isfinite = requires {
    {
        isfinite(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misfinite =
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isfinite_t, T>>) &&
    requires {
        {
            isfinite(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isfinite_t, T>>;
    };

template <>
struct extended_impl<isfinite_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <extended_vector T>
    requires unqualified_extended_isfinite<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isfinite(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misfinite<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isfinite(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_misfinite<cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(cmask_t<T, M> cmask, T&& arg) {
        return isfinite(cmask, __DPL forward<T>(arg));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::isfinite_t isfinite{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
