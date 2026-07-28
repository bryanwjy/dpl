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

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isnormal(...) noexcept = delete;

struct DPL_EMPTY_BASES isnormal_t :
    private math_operation_base<isnormal_t>,
    private maskable_predicate_base<isnormal_t> {
    using math_operation_base<isnormal_t>::operator();
    using maskable_predicate_base<isnormal_t>::operator();
};

template <>
struct operation_signature<isnormal_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isnormal_t> {
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
    requires floating_point_like<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        auto const exp = dx::bwand(val, dx::infinity);
        return dx::cmpneq(exp, dx::infinity) && dx::cmpgt(exp, dx::zero);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point_like<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_mask<E, A> mask, basic_vector<E, A> val) noexcept {
        auto const exp = dx::bwand(val, dx::infinity);
        auto const zero = dx::broadcast<E, A>(dx::zero);
        return dx::cmpneq(exp, dx::infinity) && dx::cmpgt(mask, exp, zero);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        auto const exp = dx::bwand(val, dx::infinity);
        auto const zero = dx::broadcast<T>(dx::zero);
        return dx::cmpneq(exp, dx::infinity) && dx::cmpgt(cmask, exp, zero);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isnormal = requires {
    {
        isnormal(internal::abi<A>, internal::declarg<T>())
    } -> same_as<basic_mask<simd_element_type_t<T>, A>>;
};

template <typename S, typename T>
concept unqualified_canonical_misnormal = cpo_invocable<isnormal_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isnormal_t, T>>) && requires {
        {
            isnormal(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isnormal_t, T>>;
    };

template <>
struct canonical_impl<isnormal_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpeq_t, T>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_isnormal<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T arg) noexcept {
        return isnormal(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misnormal<mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(mask_t<T> src, T val) noexcept {
        return isnormal(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misnormal<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(M cmask, T val) noexcept {
        return isnormal(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isnormal = requires {
    {
        isnormal(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misnormal = cpo_invocable<isnormal_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isnormal_t, T>>) &&
    requires {
        {
            isnormal(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isnormal_t, T>>;
    };

template <>
struct extended_impl<isnormal_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <extended_vector T>
    requires unqualified_extended_isnormal<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isnormal(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misnormal<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isnormal(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isnormal_t, T> M>
    requires unqualified_extended_misnormal<
        launder_cmask_t<cpo_result_t<isnormal_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isnormal(dx::to_const_mask<cpo_result_t<isnormal_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isnormal_t isnormal{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
