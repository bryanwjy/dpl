// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/compare.h"
#include "dpl/core/operations/evaluate.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * @brief Indicates that a SIMD value is already normalized for direct mask
 * conversion.
 *
 * This sentinel enables a fast-path overload of `to_simd_mask` by informing the
 * implementation that the input SIMD object already satisfies the backend's
 * normalized mask representation.
 *
 * The exact definition of a normalized mask is ABI-specific. In general, it
 * means that each lane is already encoded in the precise bit pattern expected
 * by the backend's native mask type, allowing conversion to bypass per-lane
 * normalization checks (such as testing for non-zero values) and instead
 * perform a direct reinterpretation or equivalent zero-cost conversion.
 *
 * For example, for SSE backends where mask and SIMD values share the same
 * underlying representation (`__m128`), a normalized mask requires:
 *
 * - `true` lanes to contain all bits set (`0xFFFFFFFF`)
 * - `false` lanes to contain all bits cleared (`0x00000000`)
 *
 * Other ABIs may define different normalized representations depending on their
 * native mask encoding.
 *
 * Passing this sentinel when the value does not satisfy the ABI's normalization
 * requirements results in undefined semantic behavior, as the resulting mask
 * may not reflect the intended boolean lane values.
 *
 * Example (SSE):
 *
 * @code
 * __m128 x = _mm_cmpgt_ps(a, b); // Produces normalized SSE mask lanes
 *
 * auto mask = to_simd_mask(
 *     xmm::abi_tag{},
 *     assume_normalized_mask,
 *     basic_vector<float, xmm::abi_tag>{x}
 * );
 * @endcode
 *
 * In contrast, arbitrary SIMD values such as:
 *
 * @code
 * __m128 x = _mm_set_ps(1.f, 0.f, -3.f, 2.f);
 * @endcode
 *
 * are not guaranteed to be normalized and must use the regular `to_simd_mask`
 * overload.
 */
DPL_EXPORT struct assume_normalized_mask_t {
    __DPL_HIDE_FROM_ABI explicit constexpr assume_normalized_mask_t() noexcept =
        default;
};

DPL_EXPORT inline constexpr assume_normalized_mask_t assume_normalized_mask{};
} // namespace datapar

namespace datapar::internal {
void to_simd_mask(...) noexcept = delete;

struct to_simd_mask_t {
private:
    template <simd_abi A, simd_element_for<A> E>
    static constexpr basic_mask<E, A> fallback(
        basic_vector<E, A> src) noexcept {
        return dx::cmpneq(src, dx::zero_v<basic_vector<E, A>>);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> src) noexcept {
        if constexpr (requires { to_simd_mask(internal::abi<A>, src); }) {
            if consteval {
                return fallback(src);
            } else {
                return to_simd_mask(internal::abi<A>, src);
            }
        } else {
            return fallback(src);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T src) noexcept {
        if constexpr (requires {
                          {
                              to_simd_mask(src)
                          } -> mask_with_common_abi<simd_abi_type_t<T>>;
                      }) {
            return to_simd_mask(src);
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(src));
        } else {
            return operator()(dx::to_canonical(src));
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        assume_normalized_mask_t tag, basic_vector<E, A> src) noexcept {
        if constexpr (requires { to_simd_mask(internal::abi<A>, tag, src); }) {
            if consteval {
                return operator()(src);
            } else {
                return to_simd_mask(internal::abi<A>, tag, src);
            }
        } else {
            return operator()(src);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        assume_normalized_mask_t tag, T src) noexcept {
        if constexpr (requires {
                          {
                              to_simd_mask(tag, src)
                          } -> mask_with_common_abi<simd_abi_type_t<T>>;
                      }) {
            return to_simd_mask(tag, src);
        } else if constexpr (simd_expression<T>) {
            return operator()(tag, dx::evaluate(src));
        } else {
            return operator()(tag, dx::to_canonical(src));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::to_simd_mask_t to_simd_mask{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
