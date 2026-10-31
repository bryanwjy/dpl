// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/accuracy.h"
#include "dpl/core/math/fixup.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rsqrt(...) noexcept = delete;

struct DPL_EMPTY_BASES rsqrt_t :
    public math_operation_base<rsqrt_t>,
    public maskable_transform_base<rsqrt_t> {
    using math_operation_base<rsqrt_t>::operator();
    using maskable_transform_base<rsqrt_t>::operator();
};

template <>
struct operation_signature<rsqrt_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mrsqrt = cpo_invocable<rsqrt_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<rsqrt_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            rsqrt(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<rsqrt_t, T>>;
    };

template <>
struct canonical_impl<rsqrt_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { rsqrt(internal::abi<T>, val); }
    {
        return rsqrt(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrsqrt<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return rsqrt(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrsqrt<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return rsqrt(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrsqrt<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return rsqrt(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrsqrt<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return rsqrt(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_rsqrt = requires {
    { rsqrt(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mrsqrt = cpo_invocable<rsqrt_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<rsqrt_t, T>>) &&
    requires {
        {
            rsqrt(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<rsqrt_t, T>>;
    };

template <>
struct extended_impl<rsqrt_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_rsqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return rsqrt(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return rsqrt( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return rsqrt( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<rsqrt_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return rsqrt(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<rsqrt_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mrsqrt<dx::zero_t,
            launder_cmask_t<cpo_result_t<rsqrt_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return rsqrt(zero, dx::to_const_mask<cpo_result_t<rsqrt_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

struct sqrt_t;

template <>
struct fallback_impl<rsqrt_t> {
private:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T fix(T result, T val) noexcept {
        constexpr auto fixflags = fpfix::condition<fpfix::nan, fpfix::copy> //
            | fpfix::condition<fpfix::infinite, dx::zero>                   //
            | fpfix::condition<fpfix::negative, dx::nan>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(result, val, fixflags);
        } else {
            result = dx::select(dx::cmpgt(val, dx::zero), result, dx::all_bits);
            result = dx::select(dx::cmpeq(val, dx::zero), dx::infinity, result);
            result = dx::select(dx::isinf(val), dx::zero, result);
            return result;
        }
    }

    template <floating_point_like E>
    static constexpr unsigned_representation_t<E> quake_lut = [] {
        if constexpr (is_same_v<E, double>) {
            return 0x5fe6eb50c7b537a9ull;
        } else if constexpr (is_same_v<E, float>) {
            return 0x5f3759dfu;
        } else if constexpr (is_same_v<E, ext::float16>) {
            return 0x59bau;
        } else {
            static_assert(is_same_v<E, ext::bfloat16>);
            return __DPL to_underlying(
                __DPL to_bit_representation(static_cast<ext::bfloat16>(
                    __DPL bit_cast<float>(0x5f3759dfu))));
        }
    }();

    template <floating_point_like E>
    static constexpr auto prenormalizer = []() {
        if constexpr (floating_point_traits<E>::digits >=
            floating_point_traits<float>::digits) {
            return static_cast<E>(0x1.p64);
        } else {
            return static_cast<E>(0x1.p12);
        }
    }();

    template <floating_point_like E>
    static constexpr auto normalizer = []() {
        if constexpr (floating_point_traits<E>::digits >=
            floating_point_traits<float>::digits) {
            return static_cast<E>(0x1.p32);
        } else {
            return static_cast<E>(0x1.p6);
        }
    }();

    template <canonical_vector T>
    static constexpr T refine(T y, T, immediate<0>) noexcept {
        return y;
    }

    template <canonical_vector T, int N>
    static constexpr T refine(T y, T half_x, immediate<N>) noexcept {
        static_assert(N >= 1);
        using E = simd_element_type_t<T>;
        constexpr E threehalves = 1.5;
        auto result = dx::multiply(y, y);
        result = dx::nmulsub(result, half_x, threehalves);
        result = dx::multiply(result, y);
        return refine(result, half_x, imm<N - 1>);
    }

    friend fallback_impl<sqrt_t>;

    static consteval int refine_steps(int target) noexcept {
        auto const lut_accuracy = 4.8f;
        auto const log2threehalves = 0.585f;
        int n = 0;
        for (auto b = lut_accuracy; b < target; b = 2 * b - log2threehalves) {
            ++n;
        }
        return n;
    }

    template <canonical_vector T, accuracy_tag Tag = accuracy::approximate_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T normal_rsqrt(
        T val, Tag tag = accuracy::approximate) noexcept {
        // normal: i.e. does not deal with subnormals
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
        using uint_t = unsigned_representation_t<E>;
        auto const vlut = dx::broadcast<uint_t, A>(quake_lut<E>);
        auto y = dx::reinterpret<E>(dx::subtract(
            vlut, dx::bwshift_right(dx::reinterpret<uint_t>(val), imm<1>)));
        constexpr int steps = [tag] {
            if constexpr (tag == accuracy::approximate) {
                return refine_steps(floating_point_traits<E>::digits / 2);
            } else {
                return refine_steps(floating_point_traits<E>::digits);
            }
        }();
        constexpr E half = 0.5;
        return refine(y, dx::multiply(half, val), imm<steps>);
    }

public:
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T val, accuracy::approximate_t approximate) noexcept {
        auto const x = fallback_impl::normal_rsqrt(val, approximate);
        return fallback_impl::fix(x, val);
    }

    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T val, accuracy::precise_t precise) noexcept {

        auto x = fallback_impl::normal_rsqrt(val, precise);

        return fallback_impl::fix(x, val);
    }

    template <canonical_vector T>
    requires cpo_invocable<fallback_impl, T, accuracy::approximate_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        return operator()(val, accuracy::approximate);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rsqrt_t rsqrt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
