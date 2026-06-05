// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/internal/scan.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void compress(...) noexcept = delete;

struct compress_t;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_compress = requires(S src, M mask, T val) {
    {
        compress(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_compress = requires(S src, M mask, T val) {
    {
        compress(src, mask, val)
    } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_canonical_icompress = requires(S src, R val) {
    {
        compress(internal::abi<A>, src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_type_with<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_extended_icompress = requires(S src, R val) {
    {
        compress(src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_type_with<S>;
};

struct compress_t {
private:
    template <typename S, typename M, typename T>
    requires (scalable_mask<M> || scalable_mask<T> ||
        scalable_abi<common_abi_t<M, T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        S src, M mask, T val) noexcept {
        using A = common_abi_t<M, T>;
        using I = signed_representation_t<simd_element_type_t<M>>;
        auto const simd_size = simd_abi_traits<I, A>::size();
        auto const idx = dx::lane_index<I, A>();
        auto rank = exscan_sum_base::operator()(mask);
        for (auto i = 1zu; i < simd_size; i <<= 1) {
            auto const dist = dx::broadcast<I, A>(static_cast<I>(i));
            auto const perm = idx ^ dist;
            auto const perm_val = dx::permute(val, perm);
            auto const perm_rank = dx::permute(rank, perm);

            auto const move =
                (rank & dist) != dx::zero && perm_rank == (rank ^ dist);
            val = dx::select(move, perm_val, val);
            rank = dx::select(move, perm_rank, rank);
        }

        return dx::select(idx < dx::popcount(mask), val, src);
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        S src, M mask, T val) noexcept {
        using A = common_abi_t<M, T>;
        using I = signed_representation_t<simd_element_type_t<M>>;
        auto const rank = exscan_sum_base::operator()(mask);
        [&val]<size_t J>(this auto self, auto rank, immediate<J>) {
            constexpr auto butterfly = []<size_t... Is>(index_sequence<Is...>) {
                return index_sequence<(Is ^ J)...>{};
            }(iota_sequence<I, A>);
            auto const perm_val = dx::permute(val, butterfly);
            auto const perm_rank = dx::permute(rank, butterfly);
            auto const dist = dx::broadcast<I, A>(J);
            auto const move =
                (rank & dist) != dx::zero && perm_rank == (rank ^ dist);
            val = dx::select(move, perm_val, val);
            if constexpr (J < simd_abi_traits<T>::size) {
                self(dx::select(move, perm_rank, rank), imm<J * 2>);
            }
        }(rank, imm<1zu>);

        return dx::select(
            dx::lane_index<I, A>() < dx::popcount(mask), val, src);
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        S src, M cmask, T val) noexcept {
        using A = common_abi_t<canonical_if_zero_t<S, T>, T>;
        using I = signed_representation_t<simd_element_type_t<T>>;
        constexpr auto rank =
            exscan_sum_base::operator()(basic_mask<I, A>(cmask));
        [&val]<size_t J>(this auto self, auto rank, immediate<J>) {
            constexpr auto idx = []<size_t... Is>(index_sequence<Is...>) {
                return index_sequence<(Is ^ J)...>{};
            }(iota_sequence<I, A>);

            auto const perm_val = dx::permute(val, idx);
            auto const perm_rank = dx::permute(rank, idx);
            auto const dist = dx::broadcast<I, A>(J);
            auto const move =
                (rank & dist) != dx::zero && perm_rank == (rank ^ dist);
            val = dx::select(move, perm_val, val);
            if constexpr (J <
                simd_abi_traits<canonical_if_zero_t<S, T>>::size) {
                self(dx::select(move, perm_rank, rank), imm<J * 2>);
            }
        }(rank, imm<1zu>);

        constexpr auto V = (1zu << dx::popcount(cmask)) - 1;
        return dx::selecti<V>(val, src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_element_type_t<T>, typename M::abi_type>;

public:
    template <canonical_vector S, canonical_mask M, canonical_vector T>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_compress<S, M, T>) {
                if consteval {
                    return compress_t::fallback(src, mask, val);
                } else {
                    return compress(internal::abi<A>, src, mask, val);
                }
            } else {
                return compress_t::fallback(src, mask, val);
            }
        } else if constexpr (unqualified_canonical_compress<S, M, T>) {
            return compress(internal::abi<A>, src, mask, val);
        } else {
            return compress_t::fallback(src, mask, val);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_compress<S, M, T>) {
            return compress(src, mask, val);
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return compress_t::fallback(src, mask, val);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_compress<zero_t, M, T>) {
                if consteval {
                    return operator()(dx::broadcast<S>(dx::zero), mask, val);
                } else {
                    return compress(internal::abi<A>, dx::zero, mask, val);
                }
            } else {
                return operator()(dx::broadcast<S>(dx::zero), mask, val);
            }
        } else if constexpr (unqualified_canonical_compress<zero_t, M, T>) {
            return compress(internal::abi<A>, dx::zero, mask, val);
        } else {

            return operator()(dx::broadcast<S>(dx::zero), mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_compress<zero_t, M, T>) {
            return compress(dx::zero, mask, val);
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            using S = broadcast_type<canonical_type_t<M>, canonical_type_t<T>>;
            return operator()(dx::broadcast<S>(dx::zero), mask, mask);
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<compress_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_icompress<S, M, T>) {
                if consteval {
                    return compress_t::fallbacki(src, cmask, val);
                } else {
                    return compress(internal::abi<A>, src, cmask, val);
                }
            } else {
                return compress_t::fallbacki(src, cmask, val);
            }
        } else if constexpr (unqualified_canonical_icompress<S, M, T>) {
            return compress(internal::abi<A>, src, cmask, val);
        } else {
            return compress_t::fallbacki(src, cmask, val);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_icompress<S, M, T>) {
            return compress(src, cmask, val);
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            simd_expression<T>) {
            return operator()(dx::evaluate(src), mask, dx::evaluate(val));
        } else {
            return compress_t::fallbacki(src, cmask, val);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<T>;
        if constexpr (unqualified_canonical_icompress<zero_t, M, T>) {
            if consteval {
                return operator()(dx::broadcast<T>(dx::zero), mask, val);
            } else {
                constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
                return compress(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return operator()(dx::broadcast<T>(dx::zero), mask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_icompress<zero_t, M, T>) {
            return compress(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(mask, dx::evaluate(val));
        } else {
            return operator()(
                dx::broadcast<canonical_type_t<T>>(dx::zero), mask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T> && invocable<compress_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::compress_t compress{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END