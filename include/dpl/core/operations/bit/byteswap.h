// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bit/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/byteswap.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void byteswap(...) noexcept = delete;
template <auto>
void byteswap(...) noexcept = delete;

struct byteswap_t;

template <typename T, typename U>
concept canonical_byteswap_result =
    simd_vector<T> && same_as<typename T::value_type, typename U::value_type> &&
    same_as<typename T::abi_type, typename U::abi_type>;

template <typename T, typename U>
concept extended_byteswap_result =
    simd_vector<T> && same_as<typename T::value_type, typename U::value_type> &&
    common_abi_with<typename T::abi_type, typename U::abi_type>;

template <typename T>
concept unqualified_canonical_byteswap = requires(T val) {
    { byteswap(internal::abi<T>, val) } -> canonical_byteswap_result<T>;
};

template <typename T>
concept unqualified_extended_byteswap = requires(T val) {
    { byteswap(val) } -> extended_byteswap_result<T>;
};

template <typename T>
concept unqualified_byteswap = unqualified_extended_byteswap<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<byteswap_t, canonical_type_t<T>>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mbyteswap = requires(S src, C mask, T val) {
    {
        byteswap(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mbyteswap = requires(S src, C mask, T val) {
    {
        byteswap(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept decayable_mbyteswap = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                  operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(byteswap_t op, canonical_or_zero_t<S, T, A> s,
        canonical_type_t<C> c, canonical_type_t<T> t) { op(s, c, t); };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept extended_mbyteswap = unqualified_extended_mbyteswap<S, C, T, A> ||
    decayable_mbyteswap<S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbyteswap = requires(S src, M mask, T val) {
    {
        byteswap(internal::abi<A>, src,
            internal::to_const_mask<A, byteswap_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbyteswap = requires(S src, M mask, T val) {
    {
        byteswap(src, internal::to_const_mask<A, byteswap_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbyteswap = decayable_vector_for<canonical_if_zero_t<S, T>,
                                   operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(byteswap_t op, canonical_or_zero_t<S, T, A> s, M mask,
        canonical_type_t<T> t) { op(s, mask, t); };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbyteswap = unqualified_extended_imbyteswap<S, M, T, A> ||
    decayable_imbyteswap<S, M, T, A>;

struct byteswap_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> arg) noexcept {
        using ubit = unsigned_representation_t<E>;
        return internal::transform<basic_vector<ubit, A>>(
            [](auto val) {
                auto const count = __DPL byteswap(__DPL bit_cast<ubit>(val));
                return static_cast<ubit>(count);
            },
            arg);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_byteswap<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return byteswap(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !integral<E>) &&
        unqualified_canonical_byteswap<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return byteswap(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_byteswap<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_byteswap<T>) {
            return byteswap(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mbyteswap<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(pass, mask, val);
            } else {
                return byteswap(internal::abi<A>, pass, mask, val);
            }
        } else {
            return internal::masked<byteswap_t>(pass, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2> ||
                 !integral<E>) &&
        simd_element_for<E, A2> &&
        maskable_args<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>> &&
        unqualified_canonical_mbyteswap<basic_vector<E, A1>,
            basic_mask<MaskE, A1>, basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A1> operator()(basic_vector<E, A1> pass,
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return byteswap(internal::abi<A1>, pass, mask, val);
    }

    template <simd_vector Pass, simd_mask Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<Pass, Mask, Arg> &&
        extended_mbyteswap<byteswap_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Pass pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mbyteswap<Pass, Mask, Arg>) {
            return byteswap(pass, mask, arg);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mbyteswap<zero_t,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(mask, val);
            } else {
                return byteswap(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2> ||
                 !integral<E>) &&
        simd_element_for<E, A2> &&
        zmaskable_args<basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mbyteswap<dx::zero_t, basic_mask<MaskE, A1>,
            basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return byteswap(internal::abi<A1>, dx::zero, mask, val);
    }

    template <simd_mask Mask, simd_vector Arg>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_mbyteswap<byteswap_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mbyteswap<dx::zero_t, Mask, Arg>) {
            return byteswap(mask, arg);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(arg));
        }
    }

    template <simd_mask Mask, simd_vector Arg>
    requires requires(Mask mask, Arg arg) { byteswap_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> pass, Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbyteswap<basic_vector<E, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(pass, mask, val);
            } else {
                return byteswap(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<byteswap_t>(pass, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi InA>
    requires (different_from<SA, InA> || scalable_abi<SA> ||
                 scalable_abi<InA> || !integral<E>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, InA>> &&
        unqualified_canonical_imbyteswap<basic_vector<E, SA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        Mask mask, basic_vector<E, InA> val) noexcept {
        return byteswap(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbyteswap<byteswap_t, S, Mask, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, Mask mask, T arg) noexcept {
        if constexpr (unqualified_extended_mbyteswap<S, Mask, T>) {
            return byteswap(pass, dx::to_compatible_const_mask<S>(mask), arg);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbyteswap<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(mask, val);
            } else {
                return byteswap(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask>
    requires (scalable_abi<InA> || !integral<E>) && simd_element_for<E, InA> &&
        imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imbyteswap<zero_t, Mask, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        Mask mask, basic_vector<E, InA> val) noexcept {
        return byteswap(internal::abi<InA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, InA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> Mask>
    requires imm_zmaskable_args<T> &&
        extended_imbyteswap<byteswap_t, zero_t, Mask, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, T arg) noexcept {
        if constexpr (unqualified_extended_imbyteswap<zero_t, Mask, T>) {
            return byteswap(
                dx::zero, dx::to_compatible_const_mask<T>(mask), arg);
        } else {
            return operator()(
                dx::to_compatible_const_mask<T>(mask), dx::to_canonical(arg));
        }
    }

    template <simd_vector T, const_mask_for<T> Mask>
    requires requires(Mask mask, T arg) { byteswap_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, T arg) noexcept {
        return operator()(mask, arg);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::byteswap_t byteswap{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
