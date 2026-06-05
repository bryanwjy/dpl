// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void splice(...) noexcept = delete;

struct splice_t;

template <typename M, typename L, typename R,
    typename A =
        common_abi_t<L, canonical_if_zero_t<R, L, common_abi_t<L, M>>, M>>
concept unqualified_canonical_splice = requires(M mask, L lhs, R rhs) {
    {
        splice(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_select_vector<M, L, R, A>;
};

template <typename M, typename L, typename R,
    typename A =
        common_abi_t<L, canonical_if_zero_t<R, L, common_abi_t<L, M>>, M>>
concept unqualified_extended_splice = requires(M mask, L lhs, R rhs) {
    { splice(mask, lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, canonical_if_zero_t<R, L>>>
concept unqualified_canonical_splicei = requires(L lhs, R rhs) {
    {
        splice(internal::abi<A>, internal::select_mask<M, L, R>(), lhs, rhs)
    } -> canonical_selecti_vector<L, R, A>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, canonical_if_zero_t<R, L>>>
concept unqualified_extended_splicei = requires(L lhs, R rhs) {
    {
        splice(internal::select_mask<M, L, R>(), lhs, rhs)
    } -> vector_with_common_abi<A>;
};

struct splice_t {
private:
    template <typename M, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        M mask, L lhs, R rhs) noexcept {
        auto const low = dx::countr_zero(mask);
        auto const high = dx::countl_zero(mask);
        return dx::slide_left(dx::shift_right(lhs, high), rhs, high + low);
    }

    template <typename M, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        M cmask, L lhs, R rhs) noexcept {
        constexpr auto low = dx::countr_zero(cmask);
        constexpr auto high = dx::countr_zero(cmask);
        return dx::slide_left(
            dx::shift_right(lhs, imm<high>), rhs, imm<high + low>);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_element_type_t<T>, typename M::abi_type>;

public:
    template <canonical_mask M, canonical_vector L, canonical_vector R>
    requires maskable_args<L, M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R, M>;
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<R>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_splice<L, M, R>) {
                if consteval {
                    return splice_t::fallback(mask, lhs, rhs);
                } else {
                    return splice(internal::abi<A>, mask, lhs, rhs);
                }
            } else {
                return splice_t::fallback(mask, lhs, rhs);
            }
        } else if constexpr (unqualified_canonical_splice<L, M, R>) {
            return splice(internal::abi<A>, mask, lhs, rhs);
        } else {
            return splice_t::fallback(mask, lhs, rhs);
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        maskable_args<L, M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_splice<M, L, R>) {
            return splice(mask, lhs, rhs);
        } else if constexpr (simd_expression<M> || simd_expression<L> ||
            simd_expression<R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return splice_t::fallback(mask, lhs, rhs);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, T val, dx::zero_t tag = dx::zero) noexcept {
        using A = simd_abi_type_t<M>;
        using R = broadcast_type<M, T>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_splice<M, T, dx::zero_t>) {
                if consteval {
                    return operator()(mask, val, dx::broadcast<R>(tag));
                } else {
                    return splice(internal::abi<A>, mask, val, tag);
                }
            } else {
                return operator()(mask, val, dx::broadcast<R>(tag));
            }
        } else if constexpr (unqualified_canonical_splice<M, T, dx::zero_t>) {
            return splice(internal::abi<A>, mask, val, tag);
        } else {

            return operator()(mask, val, dx::broadcast<R>(tag));
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, T val, dx::zero_t tag = dx::zero) noexcept {
        if constexpr (unqualified_extended_splice<M, T, dx::zero_t>) {
            return splice(mask, val, tag);
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            using R = broadcast_type<M, T>;
            return operator()(mask, val, dx::broadcast<R>(tag));
        }
    }

    template <typename M, canonical_vector L, canonical_vector R>
    requires const_mask_for<M, L> && const_mask_for<M, R> &&
        imm_maskable_args<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        static_assert(fixed_width_vector<L> && fixed_width_vector<R>);
        constexpr auto cmask = [](M mask) {
            if constexpr (simd_abi_traits<L>::size > simd_abi_traits<R>::size) {
                return dx::to_compatible_const_mask<L>(mask);
            } else {
                return dx::to_compatible_const_mask<R>(mask);
            }
        }(mask);

        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<R>>) {
            if constexpr (unqualified_canonical_splicei<M, L, R>) {
                if consteval {
                    return splice_t::fallbacki(cmask, lhs, rhs);
                } else {
                    return splice(internal::abi<A>, cmask, lhs, rhs);
                }
            } else {
                return splice_t::fallbacki(cmask, lhs, rhs);
            }
        } else if constexpr (unqualified_canonical_splicei<M, L, R>) {
            return splice(internal::abi<A>, cmask, lhs, rhs);
        } else {
            return splice_t::fallbacki(cmask, lhs, rhs);
        }
    }

    template <typename M, simd_vector L, simd_vector R>
    requires const_mask_for<M, L> && const_mask_for<M, R> &&
        (extended_vector<L> || extended_vector<R>) && imm_maskable_args<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        static_assert(fixed_width_vector<L> && fixed_width_vector<R>);
        constexpr auto cmask = [](M mask) {
            if constexpr (simd_abi_traits<L>::size > simd_abi_traits<R>::size) {
                return dx::to_compatible_const_mask<L>(mask);
            } else {
                return dx::to_compatible_const_mask<R>(mask);
            }
        }(mask);
        if constexpr (unqualified_extended_splice<M, L, R>) {
            return splice(cmask, lhs, rhs);
        } else if constexpr (simd_expression<L> || simd_expression<R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return splice_t::fallback(cmask, lhs, rhs);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, T val, dx::zero_t tag = dx::zero) noexcept {
        using A = simd_abi_type_t<T>;
        if constexpr (unqualified_canonical_splicei<M, T, dx::zero_t>) {
            if consteval {
                return operator()(mask, val, dx::broadcast<T>(tag));
            } else {
                constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
                return splice(internal::abi<A>, cmask, val, tag);
            }
        } else {
            return operator()(mask, val, dx::broadcast<T>(tag));
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        M mask, T val, dx::zero_t tag = dx::zero) noexcept {
        if constexpr (unqualified_extended_splicei<M, T, dx::zero_t>) {
            return splice(dx::to_compatible_const_mask<T>(mask), val, tag);
        } else if constexpr (simd_expression<T>) {
            return operator()(mask, dx::evaluate(val));
        } else {
            return operator()(mask, val, dx::broadcast<T>(tag));
        }
    }
};

template <auto V>
struct splicei_t {};

template <integral auto V>
struct splicei_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <typename L, simd_vector R>
    requires requires {
        typename canonical_if_zero_t<R, L>;
        typename mask_type<canonical_if_zero_t<R, L>>;
        requires regular_invocable<splice_t, L,
            mask_type<canonical_if_zero_t<R, L>>, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<canonical_if_zero_t<R, L>> cmask{};
        return splice_t::operator()(cmask, lhs, rhs);
    }

    template <simd_vector T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<splice_t, mask_type<T>, T, dx::zero_t>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        constexpr mask_type<T> cmask{};
        return splice_t::operator()(cmask, val, dx::zero);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
DPL_EXPORT inline constexpr internal::splice_t splice{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
