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
#  include "dpl/core/immediate/constants/msb.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isnan(...) noexcept = delete;

struct DPL_EMPTY_BASES isnan_t :
    private math_operation_base<isnan_t>,
    private maskable_predicate_base<isnan_t> {
    using math_operation_base<isnan_t>::operator();
    using maskable_predicate_base<isnan_t>::operator();
};

template <>
struct operation_signature<isnan_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isnan_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        using uint = unsigned_representation_t<E>;
        auto const inf_bits =
            dx::reinterpret<uint>(dx::broadcast<E, A>(dx::infinity));
        auto const abs_val = dx::bwandnot(dx::reinterpret<uint>(val), dx::msb);
        return dx::cmpgt(abs_val, inf_bits);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_mask<E, A> mask, basic_vector<E, A> val) noexcept {
        using uint = unsigned_representation_t<E>;
        auto const inf_bits =
            dx::reinterpret<uint>(dx::broadcast<E, A>(dx::infinity));
        auto const abs_val = dx::bwandnot(dx::reinterpret<uint>(val), dx::msb);
        return dx::cmpgt(mask, abs_val, inf_bits);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        using uint = unsigned_representation_t<simd_element_type_t<T>>;
        auto const inf_bits =
            dx::reinterpret<uint>(dx::broadcast<T>(dx::infinity));
        auto const abs_val = dx::bwandnot(dx::reinterpret<uint>(val), dx::msb);
        return dx::cmpgt(cmask, abs_val, inf_bits);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isnan = requires {
    {
        isnan(internal::abi<A>, internal::declarg<T>())
    } -> same_as<basic_mask<simd_element_type_t<T>, A>>;
};

template <typename S, typename T>
concept unqualified_canonical_misnan = cpo_invocable<isnan_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isnan_t, T>>) && requires {
        {
            isnan(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isnan_t, T>>;
    };

template <>
struct canonical_impl<isnan_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpeq_t, T>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_isnan<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T arg) noexcept {
        return isnan(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misnan<mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(mask_t<T> src, T val) noexcept {
        return isnan(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misnan<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(M cmask, T val) noexcept {
        return isnan(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isnan = requires {
    {
        isnan(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misnan = cpo_invocable<isnan_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isnan_t, T>>) &&
    requires {
        {
            isnan(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isnan_t, T>>;
    };

template <>
struct extended_impl<isnan_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <extended_vector T>
    requires unqualified_extended_isnan<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isnan(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misnan<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isnan(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isnan_t, T> M>
    requires unqualified_extended_misnan<
        launder_cmask_t<cpo_result_t<isnan_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isnan(dx::to_const_mask<cpo_result_t<isnan_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isnan_t isnan{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
