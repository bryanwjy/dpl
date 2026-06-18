// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/transform.h"
#include "dpl/core/operations/pack_mask.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void countr_one(...) noexcept = delete;

struct countr_one_t :
    private bit_manipulation_base<countr_one_t>,
    private maskable_transform_base<countr_one_t> {
    using operation_base<countr_one_t>::operator();
    using maskable_transform_base<countr_one_t>::operator();

    template <const_mask_like M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t operator()(M) noexcept {
        return __DPL countr_one(M::value);
    }
};

template <>
struct operation_signature<countr_one_t> {
    template <simd_type T>
    static consteval void operator()(T&&) noexcept {}
};

template <>
struct fallback_impl<countr_one_t> {
    template <fixed_width_abi A, simd_element_for<A> E>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<E, A> val) noexcept {
        using ubit = unsigned_representation_t<E>;
        return internal::transform<basic_vector<ubit, A>>(val, [](auto val) {
            auto const count = __DPL countr_one(__DPL to_unsigned(val));
            return static_cast<ubit>(count);
        });
    }

    template <canonical_mask T>
    requires fixed_width_mask<T> && cpo_invocable<pack_mask_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr size_t DPL_VECTORCALL operator()(T val) noexcept {
        return __DPL countr_one(dx::pack_mask(val));
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mcountr_one = cpo_invocable<countr_one_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<countr_one_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            countr_one(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<countr_one_t, T>>;
    };

template <>
struct canonical_impl<countr_one_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG = conditional_t<simd_mask<T>, size_t,
        basic_vector<unsigned_representation_t<simd_element_type_t<T>>,
            simd_abi_type_t<T>>>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <canonical_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T val) noexcept
    requires requires { countr_one(internal::abi<T>, val); }
    {
        return countr_one(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcountr_one<result_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, mask_t<T> mask, T val) noexcept {
        return countr_one(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcountr_one<result_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, M cmask, T val) noexcept {
        return countr_one(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcountr_one<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return countr_one(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcountr_one<dx::zero_t,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, M cmask, T val) noexcept {
        return countr_one(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_vector_countr_one = requires {
    { countr_one(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename T>
concept unqualified_extended_mask_countr_one = requires {
    { countr_one(internal::declarg<T>()) } -> core_convertible_to<size_t>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mcountr_one = cpo_invocable<countr_one_t, T> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<countr_one_t, T>>) &&
    requires {
        {
            countr_one(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<
            conditional_t<simd_type<S>, S, cpo_result_t<countr_one_t, T>>>;
    };

template <>
struct extended_impl<countr_one_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_vector_countr_one<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return countr_one(__DPL forward<T>(val));
    }

    template <extended_mask T>
    requires unqualified_extended_mask_countr_one<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T&& val) {
        return countr_one(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcountr_one<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return countr_one( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcountr_one<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return countr_one( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<countr_one_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcountr_one<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return countr_one(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<countr_one_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mcountr_one<dx::zero_t,
            launder_cmask_t<cpo_result_t<countr_one_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return countr_one(zero,
            dx::to_const_mask<cpo_result_t<countr_one_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::countr_one_t countr_one{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
