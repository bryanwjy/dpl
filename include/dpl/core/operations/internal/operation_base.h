// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename L, typename R>
concept only_unqualified = !canonical_simd_type<L> || !canonical_simd_type<R> ||
    !same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>>;

template <typename T, typename A, typename... Args>
concept implements_native = requires(Args... args) {
    internal::abi<A>;
    T::native(internal::abi<A>, args...);
};

template <typename T>
struct binary_operation_base {
private:
    template <typename L, typename R>
    static consteval auto selective_abi() noexcept {
        if constexpr (simd_type<L>) {
            return typename L::abi_type{};
        } else {
            return typename R::abi_type{};
        }
    }

    template <simd_type L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<L> const& arg) noexcept {
        return (arg);
    }

    template <typename L, simd_type R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<R> const& arg) noexcept {
        return (arg);
    }

    template <typename L, simd_type R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<L> arg) noexcept {
        return dx::broadcast<R>(arg);
    }

    template <simd_type L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<R> arg) noexcept {
        return dx::broadcast<L>(arg);
    }

public:
    template <typename L, typename R>
    requires (simd_type<L> && !simd_type<R> && broadcastable_to<R, L>) ||
        (simd_type<R> && !simd_type<L> && broadcastable_to<L, R>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(L lhs, R rhs) noexcept(
            (!simd_type<L> || canonical_simd_type<L>) &&
            (!simd_type<R> || canonical_simd_type<R>))
    requires requires(
        T impl) { impl(selective_cast<L, R>(lhs), selective_cast<L, R>(rhs)); }
    {
        using A = decltype(selective_abi<L, R>());
        if constexpr (implements_native<T, A, L, R>) {
            if constexpr (canonical_simd_type<L> || canonical_simd_type<R>) {
                if consteval {
                    return T::operator()(
                        selective_cast<L, R>(lhs), selective_cast<L, R>(rhs));
                } else {
                    return T::native(internal::abi<A>, lhs, rhs);
                }
            } else {
                return T::native(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_simd_type<L> || canonical_simd_type<R>) {
            return T::operator()(
                selective_cast<L, R>(lhs), selective_cast<L, R>(rhs));
        } else if constexpr (simd_type<L>) {
            return operator()(dx::to_canonical(lhs), rhs);
        } else {
            return operator()(rhs, dx::to_canonical(rhs));
        }
    }
};

template <typename T>
struct ternary_operation_base {
private:
    template <typename A, typename B, typename C>
    static consteval auto selective_abi() noexcept {
        if constexpr (simd_type<A>) {
            if constexpr (simd_type<B>) {
                return common_abi_t<typename A::abi_type,
                    typename B::abi_type>{};
            } else if constexpr (simd_type<C>) {
                return common_abi_t<typename A::abi_type,
                    typename C::abi_type>{};
            } else {
                return typename A::abi_type{};
            }
        } else if constexpr (simd_type<B>) {
            if constexpr (simd_type<C>) {
                return common_abi_t<typename B::abi_type,
                    typename C::abi_type>{};
            } else {
                return typename B::abi_type{};
            }
        } else {
            return typename C::abi_type{};
        }
    }

    template <simd_type A, typename B, typename C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<A> const& arg) noexcept {
        return (arg);
    }

    template <simd_type A, typename B, typename C>
    requires (!simd_type<B>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto selective_cast(type_identity_t<B> arg) noexcept {
        if constexpr (simd_type<C>) {
            return dx::broadcast<simd_element_type_t<A>, common_abi_t<A, C>>(
                arg);
        } else {
            return dx::broadcast<simd_element_type_t<A>, typename A::abi_type>(
                arg);
        }
    }

    template <simd_type A, typename B, typename C>
    requires (!simd_type<C>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto selective_cast(type_identity_t<C> arg) noexcept {
        if constexpr (simd_type<B>) {
            return dx::broadcast<simd_element_type_t<A>, common_abi_t<A, B>>(
                arg);
        } else {
            return dx::broadcast<simd_element_type_t<A>, typename A::abi_type>(
                arg);
        }
    }

    template <typename A, simd_type B, typename C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<B> const& arg) noexcept {
        return (arg);
    }

    template <typename A, simd_type B, typename C>
    requires (!simd_type<A>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto selective_cast(type_identity_t<A> arg) noexcept {
        if constexpr (simd_type<C>) {
            return dx::broadcast<simd_element_type_t<B>, common_abi_t<B, C>>(
                arg);
        } else {
            return dx::broadcast<simd_element_type_t<B>, typename B::abi_type>(
                arg);
        }
    }

    template <typename A, typename B, simd_type C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) selective_cast(
        type_identity_t<C> const& arg) noexcept {
        return (arg);
    }

    template <typename L, typename R>
    static consteval auto combine() noexcept {
        if constexpr (simd_vector<L> && simd_vector<L>) {
            return basic_vector<typename L::value_type, common_abi_t<L, R>>{};
        } else {
            return basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>{};
        }
    }

    template <typename S, typename L, typename R>
    using combine_t =
        rebind_simd_t<S, simd_element_type_t<S>, common_abi_t<L, R>>;

public:
    template <typename AT, typename BT, typename CT>
    requires (simd_type<AT> && !simd_type<BT> && !simd_type<CT> &&
                 broadcastable_to<BT, AT> && broadcastable_to<CT, AT>) ||
        (simd_type<AT> && simd_type<BT> && common_abi_with<AT, BT> &&
            !simd_type<CT> && broadcastable_to<CT, combine_t<CT, AT, BT>>) ||
        (simd_type<AT> && simd_type<CT> && common_abi_with<AT, CT> &&
            !simd_type<BT> && broadcastable_to<BT, combine_t<BT, AT, CT>>) ||
        (simd_type<BT> && !simd_type<AT> && !simd_type<CT> &&
            broadcastable_to<AT, BT> && broadcastable_to<CT, BT>) ||
        (simd_type<BT> && simd_type<CT> && common_abi_with<BT, CT> &&
            !simd_type<AT> && broadcastable_to<AT, combine_t<AT, BT, CT>>) ||
        (simd_type<CT> && !simd_type<AT> && !simd_type<BT> &&
            broadcastable_to<AT, CT> && broadcastable_to<BT, CT>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(AT a, BT b, CT c) noexcept(
            (!simd_type<AT> || canonical_simd_type<AT>) &&
            (!simd_type<BT> || canonical_simd_type<BT>) &&
            (!simd_type<CT> || canonical_simd_type<CT>))
    requires requires(T impl) {
        impl(selective_cast<AT, BT, CT>(a), selective_cast<AT, BT, CT>(b),
            selective_cast<AT, BT, CT>(c));
    }
    {
        using A = decltype(selective_abi<AT, BT, CT>());
        if constexpr (implements_native<T, A, AT, BT, CT>) {
            if constexpr ((canonical_simd_type<AT> || !simd_type<AT>) &&
                (canonical_simd_type<BT> || !simd_type<BT>) &&
                (canonical_simd_type<CT> || !simd_type<CT>)) {
                if consteval {
                    return T::operator()(selective_cast<AT, BT, CT>(a),
                        selective_cast<AT, BT, CT>(b),
                        selective_cast<AT, BT, CT>(c));
                } else {
                    return T::native(internal::abi<A>, a, b, c);
                }
            } else {
                return T::native(internal::abi<A>, a, b, c);
            }
        } else if constexpr ((canonical_simd_type<AT> || !simd_type<AT>) &&
            (canonical_simd_type<BT> || !simd_type<BT>) &&
            (canonical_simd_type<CT> || !simd_type<CT>)) {
            return T::operator()(selective_cast<AT, BT, CT>(a),
                selective_cast<AT, BT, CT>(b), selective_cast<AT, BT, CT>(c));
        } else if constexpr (simd_type<AT>) {
            if constexpr (simd_type<BT>) {
                return operator()(dx::to_canonical(a), dx::to_canonical(b), c);
            } else if constexpr (simd_type<CT>) {
                return operator()(dx::to_canonical(a), b, dx::to_canonical(c));
            } else {
                return operator()(dx::to_canonical(a), b, c);
            }
        } else if constexpr (simd_type<BT>) {
            if constexpr (simd_type<CT>) {
                return operator()(a, dx::to_canonical(b), dx::to_canonical(c));
            } else {
                return operator()(a, dx::to_canonical(b), c);
            }
        } else {
            return operator()(a, b, dx::to_canonical(c));
        }
    }
};

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
