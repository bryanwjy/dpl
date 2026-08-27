// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bit/common.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void popcount(...) noexcept = delete;

struct popcount_t :
    public bit_manipulation_base<popcount_t>,
    public maskable_transform_base<popcount_t> {
    using operation_base<popcount_t>::operator();
    using maskable_transform_base<popcount_t>::operator();

    template <const_mask_like M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t operator()(M) noexcept {
        return __DPL popcount(M::value);
    }
};

template <>
struct operation_signature<popcount_t> {
    template <simd_type T>
    static consteval void operator()(T&&) noexcept {}
};

template <>
struct fallback_impl<popcount_t> {
    template <canonical_vector T>
    requires fixed_width_abi<simd_abi_type_t<T>> &&
        integral<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr count_vector_t<T>
        DPL_VECTORCALL operator()(T val) noexcept {
        return internal::transform<count_vector_t<T>>(
            [](auto val) {
                using uint_t = make_unsigned_t<simd_element_type_t<T>>;
                auto const count = __DPL popcount(__DPL to_unsigned(val));
                return static_cast<uint_t>(count);
            },
            val);
    }

    template <canonical_mask T>
    requires fixed_width_mask<T> && cpo_invocable<to_bitset_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr size_t DPL_VECTORCALL operator()(T val) noexcept {
        using bitset_t = cpo_result_t<to_bitset_t, T>;
        return __DPL popcount(dx::to_bitset(val));
    }
};
template <typename T>
concept unqualified_canonical_popcount =
    requires { popcount(internal::abi<T>, internal::declarg<T>()); };

template <typename S, typename M, typename T>
concept unqualified_canonical_mpopcount = cpo_invocable<popcount_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<popcount_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            popcount(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<popcount_t, T>>;
    };

template <>
struct canonical_impl<popcount_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_popcount<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr count_vector_t<T> operator()(T val) noexcept {
        return popcount(internal::abi<T>, val);
    }

    template <canonical_mask T>
    requires unqualified_canonical_popcount<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T val) noexcept {
        return popcount(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mpopcount<count_vector_t<T>,
        simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr count_vector_t<T> operator()(
        count_vector_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return popcount(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mpopcount<count_vector_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr count_vector_t<T> operator()(
        count_vector_t<T> src, M cmask, T val) noexcept {
        return popcount(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mpopcount<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr count_vector_t<T> operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return popcount(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mpopcount<dx::zero_t, launder_cmask_t<T, M>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr count_vector_t<T> operator()(
        dx::zero_t zero, M cmask, T val) noexcept {
        return popcount(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_vector_popcount = requires {
    { popcount(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename T>
concept unqualified_extended_mask_popcount = requires {
    { popcount(internal::declarg<T>()) } -> core_convertible_to<size_t>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mpopcount = cpo_invocable<popcount_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<popcount_t, T>>) &&
    requires {
        {
            popcount(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<
            conditional_t<simd_type<S>, S, cpo_result_t<popcount_t, T>>>;
    };

template <>
struct extended_impl<popcount_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_vector_popcount<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return popcount(__DPL forward<T>(val));
    }

    template <extended_mask T>
    requires unqualified_extended_mask_popcount<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T&& val) {
        return popcount(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, vector_subsumed_by<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mpopcount<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return popcount( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, vector_subsumed_by<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mpopcount<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return popcount( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<popcount_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mpopcount<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return popcount(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<popcount_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mpopcount<dx::zero_t,
            launder_cmask_t<cpo_result_t<popcount_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return popcount(zero,
            dx::to_const_mask<cpo_result_t<popcount_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::popcount_t popcount{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
