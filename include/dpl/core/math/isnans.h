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
#  include "dpl/core/immediate/constants/mantissa_width.h"
#  include "dpl/core/immediate/constants/msb.h"
#  include "dpl/core/immediate/constants/value_bits.h"
#  include "dpl/core/operations/arithmetic/abs.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void isnans(...) noexcept = delete;

struct DPL_EMPTY_BASES isnans_t :
    private math_operation_base<isnans_t>,
    private maskable_predicate_base<isnans_t> {
    using math_operation_base<isnans_t>::operator();
    using maskable_predicate_base<isnans_t>::operator();
};

template <>
struct operation_signature<isnans_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<isnans_t> {
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
        using sint = signed_representation_t<E>;
        constexpr sint signaling_bit = sint(1) << (dx::mantissa_width_v<E> - 1);
        constexpr sint max_snan =
            __DPL bit_cast<sint>(dx::value_bits_v<E>) ^ signaling_bit;
        constexpr sint inf = __DPL bit_cast<sint>(dx::infinity_v<E>);
        using sint = signed_representation_t<E>;
        auto const abs_val = dx::reinterpret<sint>(dx::bwandnot(val, dx::msb));

        return dx::cmple(abs_val, max_snan) && dx::cmpgt(abs_val, inf);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_mask<E, A> mask, basic_vector<E, A> val) noexcept {
        using uint = unsigned_representation_t<E>;
        using sint = signed_representation_t<E>;
        constexpr sint signaling_bit = sint(1) << (dx::mantissa_width_v<E> - 1);
        constexpr sint max_snan =
            __DPL bit_cast<sint>(dx::value_bits_v<E>) ^ signaling_bit;
        constexpr sint inf = __DPL bit_cast<sint>(dx::infinity_v<E>);
        using sint = signed_representation_t<E>;
        auto const abs_val = dx::reinterpret<sint>(dx::bwandnot(val, dx::msb));
        auto const vinf = dx::broadcast<E, A>(inf);
        return dx::cmple(abs_val, max_snan) && dx::cmpgt(mask, abs_val, vinf);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(M cmask, T val) noexcept {
        using E = simd_element_type_t<T>;
        using uint = unsigned_representation_t<E>;
        using sint = signed_representation_t<E>;
        constexpr sint signaling_bit = sint(1) << (dx::mantissa_width_v<E> - 1);
        constexpr sint max_snan =
            __DPL bit_cast<sint>(dx::value_bits_v<E>) ^ signaling_bit;
        constexpr sint inf = __DPL bit_cast<sint>(dx::infinity_v<E>);
        using sint = signed_representation_t<E>;
        auto const abs_val = dx::reinterpret<sint>(dx::bwandnot(val, dx::msb));
        auto const vinf = dx::broadcast<T>(inf);
        return dx::cmple(abs_val, max_snan) && dx::cmpgt(cmask, abs_val, vinf);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_isnans = requires {
    {
        isnans(internal::abi<A>, internal::declarg<T>())
    } -> same_as<basic_mask<simd_element_type_t<T>, A>>;
};

template <typename S, typename T>
concept unqualified_canonical_misnans = cpo_invocable<isnans_t, T> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<isnans_t, T>>) && requires {
        {
            isnans(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<isnans_t, T>>;
    };

template <>
struct canonical_impl<isnans_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpeq_t, T>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_isnans<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T arg) noexcept {
        return isnans(internal::abi<T>, arg);
    }

    template <canonical_vector T>
    requires unqualified_canonical_misnans<mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(mask_t<T> src, T val) noexcept {
        return isnans(internal::abi<T>, src, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_misnans<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mask_t<T> operator()(M cmask, T val) noexcept {
        return isnans(internal::abi<T>, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_isnans = requires {
    {
        isnans(internal::declarg<T>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename T>
concept unqualified_extended_misnans = cpo_invocable<isnans_t, T> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<isnans_t, T>>) &&
    requires {
        {
            isnans(internal::declarg<S>(), internal::declarg<T>())
        } -> equivalent_mask_with<cpo_result_t<isnans_t, T>>;
    };

template <>
struct extended_impl<isnans_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <extended_vector T>
    requires unqualified_extended_isnans<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) {
        return isnans(__DPL forward<T>(arg));
    }

    template <simd_mask S, simd_vector T>
    requires (extended_mask<S> || extended_vector<T>) &&
        unqualified_extended_misnans<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, T&& arg) {
        return isnans(__DPL forward<S>(src), __DPL forward<T>(arg));
    }

    template <extended_vector T, result_cmask_for<isnans_t, T> M>
    requires unqualified_extended_misnans<
        launder_cmask_t<cpo_result_t<isnans_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, T&& arg) {
        return isnans(dx::to_const_mask<cpo_result_t<isnans_t, T>>(cmask),
            __DPL forward<T>(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::isnans_t isnans{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
