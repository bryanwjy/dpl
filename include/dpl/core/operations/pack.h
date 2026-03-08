

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/extract.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void pack(...) noexcept = delete;

struct pack_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bit_type_t<element_count<E, A>> DPL_VECTORCALL
        fallback(basic_simd_mask<E, A> arg) noexcept {
        if constexpr (convertible_to<decltype(dx::to_native_vector(arg)),
                          bit_type_t<element_count<E, A>>>) {
            return dx::to_native_vector(arg);
        } else {
            return []<size_t... Is>(
                       basic_simd_mask<E, A> arg, index_sequence<Is...>) {
                using bit = bit_type_t<element_count<E, A>>;
                return static_cast<bit>((
                    ... | (static_cast<bit>(dx::extract(arg, imm<Is>)) << Is)));
            }(arg, iota_sequence<E, A>);
        }
    }

public:
    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bit_type_t<element_count<T>> DPL_VECTORCALL operator()(
        T arg) noexcept {
        if constexpr (requires { pack(internal::abi<T>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return pack(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bit_type_t<element_count<T>> operator()(T arg) noexcept {
        if constexpr (requires { pack(internal::abi<T>, arg); }) {
            return pack(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::pack_t pack{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
