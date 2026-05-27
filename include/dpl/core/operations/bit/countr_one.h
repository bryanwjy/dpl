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
#  include "dpl/std/bit/countr.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void countr_one(...) noexcept = delete;
template <auto>
void countr_one(...) noexcept = delete;

struct countr_one_t;

template <typename T>
concept unqualified_canonical_countr_one = requires(T val) {
    {
        countr_one(internal::abi<T>, val)
    } -> canonical_vector_bit<T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_countr_one = requires(T val) {
    { countr_one(val) } -> extended_vector_bit<T, typename T::abi_type>;
};

template <typename T>
concept unqualified_countr_one = unqualified_extended_countr_one<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<countr_one_t, canonical_type_t<T>>);

template <typename T>
concept unqualified_canonical_mask_countr_one = requires(T val) {
    { countr_one(internal::abi<T>, val) } -> core_convertible_to<size_t>;
};

template <typename T>
concept unqualified_extended_mask_countr_one = requires(T val) {
    { countr_one(val) } -> core_convertible_to<size_t>;
};

template <typename T>
concept unqualified_mask_countr_one =
    unqualified_canonical_mask_countr_one<T> ||
    unqualified_extended_mask_countr_one<T> ||
    (decayable_vector_for<T, operation_category::lane_reduction> &&
        regular_invocable<countr_one_t, canonical_type_t<T>>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mcountr_one = requires(S src, C mask, T val) {
    {
        countr_one(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mcountr_one = requires(S src, C mask, T val) {
    {
        countr_one(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept decayable_mcountr_one =
    decayable_vector_for<canonical_if_zero_t<S, T, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(countr_one_t op, canonical_or_zero_t<S, T, A> s,
        canonical_type_t<C> c, canonical_type_t<T> t) { op(s, c, t); };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept extended_mcountr_one = unqualified_extended_mcountr_one<S, C, T, A> ||
    decayable_mcountr_one<S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imcountr_one = requires(S src, M mask, T val) {
    {
        countr_one(internal::abi<A>, src,
            internal::to_const_mask<A, countr_one_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imcountr_one = requires(S src, M mask, T val) {
    {
        countr_one(
            src, internal::to_const_mask<A, countr_one_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imcountr_one = decayable_vector_for<canonical_if_zero_t<S, T>,
                                     operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(countr_one_t op, canonical_or_zero_t<S, T, A> s, M mask,
        canonical_type_t<T> t) { op(s, mask, t); };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imcountr_one = unqualified_extended_imcountr_one<S, M, T, A> ||
    decayable_imcountr_one<S, M, T, A>;

struct countr_one_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_vector<E, A> arg) noexcept {
        using ubit = unsigned_representation_t<E>;
        return internal::transform<basic_vector<ubit, A>>(arg, [](auto val) {
            auto const count = __DPL countr_one(__DPL bit_cast<ubit>(val));
            return static_cast<ubit>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = static_cast<size_t>(arg[I]);
            if constexpr (I < simd_abi_traits<E, A>::size) {
                auto const mask = val ? -1zu : 0zu;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return 0;
            }
        }(arg, imm<0>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_countr_one<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return countr_one(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_countr_one<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<E, A> val) noexcept {
        return countr_one(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_countr_one<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_countr_one<T>) {
            return countr_one(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mask_countr_one<basic_mask<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return countr_one(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_mask_countr_one<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(basic_mask<E, A> val) noexcept {
        return countr_one(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_countr_one<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T val) noexcept {
        if constexpr (unqualified_extended_mask_countr_one<T>) {
            return countr_one(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<unsigned_representation_t<E>, A> pass,
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mcountr_one<
                          basic_vector<unsigned_representation_t<E>, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<countr_one_t>(pass, mask, val);
            } else {
                return countr_one(internal::abi<A>, pass, mask, val);
            }
        } else {
            return internal::masked<countr_one_t>(pass, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2>) &&
        simd_element_for<E, A2> &&
        maskable_args<basic_vector<unsigned_representation_t<E>, A1>,
            basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mcountr_one<
            basic_vector<unsigned_representation_t<E>, A1>,
            basic_mask<MaskE, A1>, basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A1> operator()(basic_vector<E, A1> pass,
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return countr_one(internal::abi<A1>, pass, mask, val);
    }

    template <simd_vector Pass, simd_mask Mask, simd_vector Arg>
    requires extended_vector_bit<Pass, Arg> &&
        (extended_vector<Pass> || extended_mask<Mask> ||
            extended_vector<Arg>) &&
        maskable_args<Pass, Mask, Arg> &&
        extended_mcountr_one<countr_one_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Pass pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mcountr_one<Pass, Mask, Arg>) {
            return countr_one(pass, mask, arg);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mcountr_one<zero_t,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<countr_one_t>(mask, val);
            } else {
                return countr_one(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2>) &&
        simd_element_for<E, A2> &&
        zmaskable_args<basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mcountr_one<dx::zero_t, basic_mask<MaskE, A1>,
            basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return countr_one(internal::abi<A1>, dx::zero, mask, val);
    }

    template <simd_mask Mask, simd_vector Arg>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_mcountr_one<countr_one_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mcountr_one<dx::zero_t, Mask, Arg>) {
            return countr_one(mask, arg);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(arg));
        }
    }

    template <simd_mask Mask, simd_vector Arg>
    requires requires(
        Mask mask, Arg arg) { countr_one_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<unsigned_representation_t<E>, A> pass, Mask mask,
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imcountr_one<
                          basic_vector<unsigned_representation_t<E>, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<countr_one_t>(pass, mask, val);
            } else {
                return countr_one(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<countr_one_t>(pass, mask, val);
        }
    }

    template <simd_abi SA, simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<unsigned_representation_t<E>, SA>> Mask>
    requires simd_element_for<unsigned_representation_t<E>, SA> &&
        (different_from<SA, InA> || scalable_abi<SA> || scalable_abi<InA>) &&
        imm_maskable_args<basic_vector<unsigned_representation_t<E>, SA>,
            basic_vector<E, InA>> &&
        unqualified_canonical_imcountr_one<
            basic_vector<unsigned_representation_t<E>, SA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, SA> operator()(
        basic_vector<unsigned_representation_t<E>, SA> pass, Mask mask,
        basic_vector<E, InA> val) noexcept {
        return countr_one(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector Arg>
    requires extended_vector_bit<S, Arg> &&
        (extended_vector<S> || extended_vector<Arg>) &&
        imm_maskable_args<S, Arg> &&
        extended_imcountr_one<countr_one_t, S, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mcountr_one<S, Mask, Arg>) {
            return countr_one(pass, dx::to_compatible_const_mask<S>(mask), arg);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imcountr_one<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<countr_one_t>(mask, val);
            } else {
                return countr_one(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <scalable_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask>
    requires imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_imcountr_one<zero_t, Mask, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, InA> operator()(
        Mask mask, basic_vector<E, InA> val) noexcept {
        return countr_one(internal::abi<InA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, InA>>(mask), val);
    }

    template <extended_vector Arg, const_mask_for<Arg> Mask>
    requires imm_zmaskable_args<Arg> &&
        extended_imcountr_one<countr_one_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_imcountr_one<zero_t, Mask, Arg>) {
            return countr_one(
                dx::zero, dx::to_compatible_const_mask<Arg>(mask), arg);
        } else {
            return operator()(
                dx::to_compatible_const_mask<Arg>(mask), dx::to_canonical(arg));
        }
    }

    template <simd_vector Arg, const_mask_for<Arg> Mask>
    requires requires(
        Mask mask, Arg arg) { countr_one_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::countr_one_t countr_one{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
