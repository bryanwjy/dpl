// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/internal/scan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void expand(...) noexcept = delete;

struct expand_t;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_expand = requires(S src, M mask, T val) {
    {
        expand(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_expand = requires(S src, M mask, T val) {
    {
        expand(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_canonical_iexpand = requires(S src, R val) {
    {
        expand(internal::abi<A>, src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_extended_iexpand = requires(S src, R val) {
    {
        expand(src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

struct expand_t {
private:
    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        S src, M mask, T val) noexcept {
        return dx::permute(src, mask, val, exscan_sum_base::operator()(mask));
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        S src, M cmask, T val) noexcept {
        using A = common_abi_t<canonical_if_zero_t<S, T>, T>;
        using I = signed_representation_t<simd_lane_type_t<T>>;
        constexpr auto rank =
            exscan_sum_base::operator()(basic_mask<I, A>(cmask));
        constexpr auto seq = []<size_t... Is>(index_sequence<Is...>) {
            return index_sequence<rank[Is]...>{};
        }(iota_sequence<I, A>);
        return dx::permute(src, cmask, val, seq);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    template <canonical_vector S, canonical_mask M, canonical_vector T>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_expand<S, M, T>) {
                if consteval {
                    return expand_t::fallback(src, mask, val);
                } else {
                    return expand(internal::abi<A>, src, mask, val);
                }
            } else {
                return expand_t::fallback(src, mask, val);
            }
        } else if constexpr (unqualified_canonical_expand<S, M, T>) {
            return expand(internal::abi<A>, src, mask, val);
        } else {
            return expand_t::fallback(src, mask, val);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_expand<S, M, T>) {
            return expand(src, mask, val);
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            simd_expression<T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return expand_t::fallback(src, mask, val);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_expand<zero_t, M, T>) {
                if consteval {
                    return expand_t::fallback(dx::zero, mask, val);
                } else {
                    return expand(internal::abi<A>, dx::zero, mask, val);
                }
            } else {
                return expand_t::fallback(dx::zero, mask, val);
            }
        } else if constexpr (unqualified_canonical_expand<zero_t, M, T>) {
            return expand(internal::abi<A>, dx::zero, mask, val);
        } else {

            return expand_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_expand<zero_t, M, T>) {
            return expand(dx::zero, mask, val);
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return expand_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<expand_t, M, T>
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
            if constexpr (unqualified_canonical_iexpand<S, M, T>) {
                if consteval {
                    return expand_t::fallbacki(src, cmask, val);
                } else {
                    return expand(internal::abi<A>, src, cmask, val);
                }
            } else {
                return expand_t::fallbacki(src, cmask, val);
            }
        } else if constexpr (unqualified_canonical_iexpand<S, M, T>) {
            return expand(internal::abi<A>, src, cmask, val);
        } else {
            return expand_t::fallbacki(src, cmask, val);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_iexpand<S, M, T>) {
            return expand(src, cmask, val);
        } else if constexpr (simd_expression<S> || simd_expression<T>) {
            return operator()(dx::evaluate(src), cmask, dx::evaluate(val));
        } else {
            return expand_t::fallbacki(src, cmask, val);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_iexpand<zero_t, M, T>) {
            if consteval {
                return expand_t::fallbacki(dx::zero, cmask, val);
            } else {
                return expand(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return expand_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_extended_iexpand<zero_t, M, T>) {
            return expand(dx::zero, cmask, val);
        } else if constexpr (simd_expression<T>) {
            return operator()(cmask, dx::evaluate(val));
        } else {
            return expand_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T> && invocable<expand_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

template <auto V>
struct expandi_t {};

template <integral auto V>
struct expandi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <simd_vector S, simd_vector R>
    requires requires {
        typename mask_type<S>;
        requires regular_invocable<expand_t, S, mask_type<S>, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<S> cmask{};
        return expand_t::operator()(src, cmask, val);
    }

    template <simd_class S, typename R>
    requires (!simd_class<R>) && requires { typename mask_type<S>; } &&
        regular_invocable<expand_t, S, mask_type<S>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<S> cmask{};
        return expand_t::operator()(src, cmask, val);
    }

    template <typename S, simd_class R>
    requires (!simd_class<S>) && requires { typename mask_type<R>; } &&
        regular_invocable<expand_t, S, mask_type<R>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<R> cmask{};
        return expand_t::operator()(src, cmask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::expandi_t<V> expandi{};
DPL_EXPORT inline constexpr internal::expand_t expand{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
