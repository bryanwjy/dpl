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
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/representation.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void signbit(...) noexcept = delete;

struct DPL_EMPTY_BASES signbit_t :
    private math_operation_base<signbit_t>,
    private maskable_predicate_base<signbit_t> {
    using math_operation_base<signbit_t>::operator();
    using maskable_predicate_base<signbit_t>::operator();
};

template <>
struct operation_signature<signbit_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<signbit_t> {
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
    requires signed_integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        return dx::cmplt(val, dx::zero);
    }

    template <canonical_vector T>
    requires signed_integral<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(result_t<T> mask, T val) noexcept {
        auto const zero = dx::broadcast<T>(dx::zero);
        return dx::cmplt(mask, val, zero);
    }

    template <canonical_vector T, imask_t<T> M>
    requires signed_integral<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(cmask_t<T, M> cmask, T val) noexcept {
        auto const zero = dx::broadcast<T>(dx::zero);
        return dx::cmplt(cmask, val, zero);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        return operator()(dx::reinterpret<signed_representation_t<E>>(val));
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_mask<E, A> mask, basic_vector<E, A> val) noexcept {
        return operator()(
            mask, dx::reinterpret<signed_representation_t<E>>(val));
    }

    template <canonical_vector T, imask_t<T> M>
    requires floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(cmask_t<T, M> cmask, T val) noexcept {
        using E = simd_element_type_t<T>;
        return operator()(
            cmask, dx::reinterpret<signed_representation_t<E>>(val));
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_signbit = requires {
    {
        signbit(internal::abi<A>, internal::declarg<T>())
    } -> same_as<basic_mask<simd_element_type_t<T>, A>>;
};

template <typename S, typename T>
concept unqualified_canonical_msignbit = canonical_mask<S> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<signbit_t, T>>) && requires {
        {
            signbit(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<signbit_t, T>>;
    };

template <>
struct canonical_impl<signbit_t> {
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
    requires unqualified_canonical_signbit<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T arg) noexcept {
        return signbit(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msignbit<mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(mask_t<T> src, T val) noexcept {
        return signbit(internal::abi<T>, src, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_msignbit<cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(cmask_t<T, M> cmask, T val) noexcept {
        return signbit(internal::abi<T>, cmask, val);
    }
};

template <typename T>
concept unqualified_extended_signbit = requires {
    {
        signbit(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_msignbit =
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<signbit_t, T>>) &&
    requires {
        {
            signbit(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<signbit_t, T>>;
    };

template <>
struct extended_impl<signbit_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <extended_vector T>
    requires unqualified_extended_signbit<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return signbit(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_msignbit<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return signbit(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_msignbit<cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(cmask_t<T, M> cmask, T&& arg) {
        return signbit(cmask, __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::signbit_t signbit{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
