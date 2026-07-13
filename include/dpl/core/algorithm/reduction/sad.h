// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduction/hsum.h"
#include "dpl/core/algorithm/rotate/rotate_left.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/arithmetic/add.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void sad(...) noexcept = delete;

/**
 * @brief Sum of Absolute Differences (SAD) between two integer vectors.
 *
 * Computes the sum of the absolute differences of corresponding elements
 * in @p lhs and @p rhs. This is an `algorithm_operation` CPO and is
 * strictly non-maskable.
 *
 * @tparam E Element type of @p lhs and @p rhs. Must be an integer type
 *           satisfying `sizeof(E) < sizeof(std::ptrdiff_t)` — i.e. strictly
 *           narrower than the processor word size. This excludes, e.g.,
 *           64-bit integer element types on typical 64-bit targets, since
 *           there is no wider native integer type available to hold the
 *           accumulated result without overflow risk.
 *
 * @param lhs First operand vector.
 * @param rhs Second operand vector, same element type and width as @p lhs.
 *
 * @return A vector whose element type is the next integer width up from
 *         `E` (e.g. `int8` -> `int16`, `int16` -> `int32`,
 *         `uint8` -> `uint16`, etc.). Only **lane 0** of the returned
 *         vector holds the sum of absolute differences; the contents of
 *         all other lanes are implementation-defined and must not be
 *         relied upon. Callers that need the scalar result should extract
 *         lane 0 explicitly.
 */
struct sad_t : public reduction_base<sad_t> {
    using operation_base<sad_t>::operator();
};

template <typename T>
concept sad_vector = simd_vector<T> && integral<simd_element_type_t<T>> &&
    (sizeof(simd_element_type_t<T>) < sizeof(ptrdiff_t));

template <>
struct operation_signature<sad_t> {
    template <typename L, typename R>
    requires (sad_vector<L> && !simd_type<R> ||
        sad_vector<R> && !simd_type<L> ||
        sad_vector<L> && common_vector_with<L, R>)
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_sad = requires {
    {
        sad(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> canonical_vector;
};

template <>
struct canonical_impl<sad_t> {
private:
    template <typename E>
    static consteval auto make_result() noexcept {
        if constexpr (sizeof(E) == sizeof(int8)) {
            using ToE = conditional_t<is_signed_v<E>, int16, uint16>;
            return ToE{};
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            using ToE = conditional_t<is_signed_v<E>, int32, uint32>;
            return ToE{};
        } else {
            using ToE = conditional_t<is_signed_v<E>, int64, uint64>;
            return ToE{};
        }
    }

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using result_t DPL_NODEBUG =
        basic_vector<decltype(make_result<simd_element_type_t<T>>()),
            simd_abi_type_t<T>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_sad<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L> operator()(L lhs, R rhs) noexcept {
        return sad(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_sad<L, R, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L> operator()(L lhs, R&& rhs) noexcept {
        return sad(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_sad<L, R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<R> operator()(L&& lhs, R rhs) noexcept {
        return sad(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_sad = requires {
    {
        sad(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <>
struct extended_impl<sad_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_sad<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return sad(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_sad<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_sad<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return sad(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<sad_t> : binary_broadcasting_fallback<sad_t> {
public:
    template <simd_abi A, integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        auto const lower = dx::subtract(dx::max(lhs, rhs), dx::min(lhs, rhs));
        auto const upper =
            dx::rotate_left(lower, imm<simd_abi_traits<A, E>::size / 2zu>);
        if constexpr (sizeof(E) == sizeof(int8)) {
            using ToE = conditional_t<is_signed_v<E>, int16, uint16>;
            return dx::add(dx::hsum(dx::element_cast<ToE>(lower)),
                dx::hsum(dx::element_cast<ToE>(upper)));
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            using ToE = conditional_t<is_signed_v<E>, int32, uint32>;
            return dx::add(dx::hsum(dx::element_cast<ToE>(lower)),
                dx::hsum(dx::element_cast<ToE>(upper)));
        } else {
            using ToE = conditional_t<is_signed_v<E>, int64, uint64>;
            return dx::add(dx::hsum(dx::element_cast<ToE>(lower)),
                dx::hsum(dx::element_cast<ToE>(upper)));
        }
    }

    using binary_broadcasting_fallback<sad_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::sad_t sad{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
