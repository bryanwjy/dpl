// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/private/fwd.h"

#include "dpl/core/dispatch/broadcastable/base.h"
#include "dpl/core/dispatch/private/concepts.h"
#include "dpl/core/dispatch/private/cpo.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename L, typename R>
concept commutative_rebindable =
    common_abi_with<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
    same_as<simd_element_type_t<L>, simd_element_type_t<R>> &&
    same_as<rebind_simd_t<L, simd_element_type_t<R>, common_abi_t<L, R>>,
        rebind_simd_t<R, simd_element_type_t<L>, common_abi_t<L, R>>> &&
    same_as<rebind_simd_t<L, simd_element_type_t<L>, common_abi_t<L, R>>,
        rebind_simd_t<R, simd_element_type_t<R>, common_abi_t<L, R>>>;

template <typename A, typename B>
struct ternary_broadcast_rebind {};
template <typename A, typename B>
using ternary_broadcast_rebind_t DPL_NODEBUG =
    typename ternary_broadcast_rebind<A, B>::type;

template <simd_type A, simd_type B>
requires commutative_rebindable<remove_cvref_t<A>, remove_cvref_t<B>>
struct ternary_broadcast_rebind<A, B> {
    using type DPL_NODEBUG = rebind_simd_t<remove_cvref_t<A>,
        simd_element_type_t<A>, common_abi_t<A, B>>;
};

template <typename T, typename A, typename B>
concept ternary_broadcastable =
    commutative_rebindable<remove_cvref_t<A>, remove_cvref_t<B>> &&
    broadcastable_to<T, ternary_broadcast_rebind_t<A, B>> &&
    broadcastable_to<T, ternary_broadcast_rebind_t<B, A>>;

template <typename D>
struct ternary_broadcastable_operation :
    public broadcastable_operation_base<D> {

    template <canonical_simd_type AT, canonical_simd_type CT,
        ternary_broadcastable<AT, CT> BT>
    requires signature_compatible<D, AT, BT, CT> &&
        (fallback_cpo_invocable<D, AT, BT, CT> ||
            canonical_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT&& bval, CT cval) noexcept {
        if constexpr (fallback_cpo_invocable<D, AT, BT, CT>) {
            if constexpr (canonical_cpo_invocable<D, AT, BT, CT>) {
                if consteval {
                    return impl::fallback<D>(
                        aval, __DPL forward<BT>(bval), cval);
                } else {
                    return impl::canonical<D>(
                        aval, __DPL forward<BT>(bval), cval);
                }
            } else {
                return impl::fallback<D>(aval, __DPL forward<BT>(bval), cval);
            }
        } else {
            return impl::canonical<D>(aval, __DPL forward<BT>(bval), cval);
        }
    }

    template <canonical_simd_type AT, canonical_simd_type BT,
        ternary_broadcastable<AT, BT> CT>
    requires signature_compatible<D, AT, BT, CT> &&
        (fallback_cpo_invocable<D, AT, BT, CT> ||
            canonical_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT&& bval, CT cval) noexcept {
        if constexpr (fallback_cpo_invocable<D, AT, BT, CT>) {
            if constexpr (canonical_cpo_invocable<D, AT, BT, CT>) {
                if consteval {
                    return impl::fallback<D>(
                        aval, bval, __DPL forward<CT>(cval));
                } else {
                    return impl::canonical<D>(
                        aval, bval, __DPL forward<CT>(cval));
                }
            } else {
                return impl::fallback<D>(aval, bval, __DPL forward<CT>(cval));
            }
        } else {
            return impl::canonical<D>(aval, bval, __DPL forward<CT>(cval));
        }
    }

    template <canonical_simd_type BT, canonical_simd_type CT,
        ternary_broadcastable<BT, CT> AT>
    requires signature_compatible<D, AT, BT, CT> &&
        (fallback_cpo_invocable<D, AT, BT, CT> ||
            canonical_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT bval, CT cval) noexcept {
        if constexpr (fallback_cpo_invocable<D, AT, BT, CT>) {
            if constexpr (canonical_cpo_invocable<D, AT, BT, CT>) {
                if consteval {
                    return impl::fallback<D>(
                        __DPL forward<AT>(aval), bval, cval);
                } else {
                    return impl::canonical<D>(
                        __DPL forward<AT>(aval), bval, cval);
                }
            } else {
                return impl::fallback<D>(__DPL forward<AT>(aval), bval, cval);
            }
        } else {
            return impl::canonical<D>(__DPL forward<AT>(aval), bval, cval);
        }
    }

    template <canonical_simd_type AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires signature_compatible<D, AT, BT, CT> &&
        (fallback_cpo_invocable<D, AT, BT, CT> ||
            canonical_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (fallback_cpo_invocable<D, AT, BT, CT>) {
            if constexpr (canonical_cpo_invocable<D, AT, BT, CT>) {
                if consteval {
                    return impl::fallback<D>(aval, __DPL forward<BT>(bval),
                        __DPL forward<CT>(cval));
                } else {
                    return impl::canonical<D>(aval, __DPL forward<BT>(bval),
                        __DPL forward<CT>(cval));
                }
            } else {
                return impl::fallback<D>(
                    aval, __DPL forward<BT>(bval), __DPL forward<CT>(cval));
            }
        } else {
            return impl::canonical<D>(
                aval, __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }

    template <canonical_simd_type BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires signature_compatible<D, AT, BT, CT> &&
        (fallback_cpo_invocable<D, AT, BT, CT> ||
            canonical_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        if constexpr (fallback_cpo_invocable<D, AT, BT, CT>) {
            if constexpr (canonical_cpo_invocable<D, AT, BT, CT>) {
                if consteval {
                    return impl::fallback<D>(
                        __DPL forward<AT>(aval), bval, __DPL forward<CT>(cval));
                } else {
                    return impl::canonical<D>(
                        __DPL forward<AT>(aval), bval, __DPL forward<CT>(cval));
                }
            } else {
                return impl::fallback<D>(
                    __DPL forward<AT>(aval), bval, __DPL forward<CT>(cval));
            }
        } else {
            return impl::canonical<D>(
                __DPL forward<AT>(aval), bval, __DPL forward<CT>(cval));
        }
    }

    template <canonical_simd_type CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT>
    requires signature_compatible<D, AT, BT, CT> &&
        (fallback_cpo_invocable<D, AT, BT, CT> ||
            canonical_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT cval) noexcept {
        if constexpr (fallback_cpo_invocable<D, AT, BT, CT>) {
            if constexpr (canonical_cpo_invocable<D, AT, BT, CT>) {
                if consteval {
                    return impl::fallback<D>(
                        __DPL forward<AT>(aval), __DPL forward<BT>(bval), cval);
                } else {
                    return impl::canonical<D>(
                        __DPL forward<AT>(aval), __DPL forward<BT>(bval), cval);
                }
            } else {
                return impl::fallback<D>(
                    __DPL forward<AT>(aval), __DPL forward<BT>(bval), cval);
            }
        } else {
            return impl::canonical<D>(
                __DPL forward<AT>(aval), __DPL forward<BT>(bval), cval);
        }
    }

    ///
    template <simd_type AT, simd_type CT, ternary_broadcastable<AT, CT> BT>
    requires signature_compatible<D, AT, BT, CT> &&
        (extended_simd_type<AT> || extended_simd_type<CT>) &&
        (extended_cpo_invocable<D, AT, BT, CT> ||
            fallback_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (extended_cpo_invocable<D, AT, BT, CT>) {
            return impl::extended<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else {
            return impl::fallback<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }

    template <simd_type AT, simd_type BT, ternary_broadcastable<AT, BT> CT>
    requires signature_compatible<D, AT, BT, CT> &&
        (extended_simd_type<AT> || extended_simd_type<BT>) &&
        (extended_cpo_invocable<D, AT, BT, CT> ||
            fallback_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (extended_cpo_invocable<D, AT, BT, CT>) {
            return impl::extended<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else {
            return impl::fallback<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }

    template <simd_type BT, simd_type CT, ternary_broadcastable<BT, CT> AT>
    requires signature_compatible<D, AT, BT, CT> &&
        (extended_simd_type<BT> || extended_simd_type<CT>) &&
        (extended_cpo_invocable<D, AT, BT, CT> ||
            fallback_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (extended_cpo_invocable<D, AT, BT, CT>) {
            return impl::extended<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else {
            return impl::fallback<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }

    template <extended_simd_type AT,
        broadcastable_to<result_or_identity_t<AT>> BT,
        broadcastable_to<result_or_identity_t<AT>> CT>
    requires signature_compatible<D, AT, BT, CT> &&
        (extended_cpo_invocable<D, AT, BT, CT> ||
            fallback_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (extended_cpo_invocable<D, AT, BT, CT>) {
            return impl::extended<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else {
            return impl::fallback<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }

    template <extended_simd_type BT,
        broadcastable_to<result_or_identity_t<BT>> AT,
        broadcastable_to<result_or_identity_t<BT>> CT>
    requires signature_compatible<D, AT, BT, CT> &&
        (extended_cpo_invocable<D, AT, BT, CT> ||
            fallback_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (extended_cpo_invocable<D, AT, BT, CT>) {
            return impl::extended<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else {
            return impl::fallback<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }

    template <extended_simd_type CT,
        broadcastable_to<result_or_identity_t<CT>> AT,
        broadcastable_to<result_or_identity_t<CT>> BT>
    requires signature_compatible<D, AT, BT, CT> &&
        (extended_cpo_invocable<D, AT, BT, CT> ||
            fallback_cpo_invocable<D, AT, BT, CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (extended_cpo_invocable<D, AT, BT, CT>) {
            return impl::extended<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else {
            return impl::fallback<D>(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        }
    }
};

template <typename D>
struct ternary_broadcasting_fallback {
    template <typename AT, typename BT, typename CT>
    requires (simd_type<AT> || simd_type<BT> || simd_type<CT>) &&
        (ternary_broadcastable<AT, BT, CT> &&
            cpo_invocable<D, ternary_broadcast_rebind_t<BT, CT>, BT, CT>) ||
        (ternary_broadcastable<BT, AT, CT> &&
            cpo_invocable<D, AT, ternary_broadcast_rebind_t<AT, CT>, CT>) ||
        (ternary_broadcastable<CT, AT, BT> &&
            cpo_invocable<D, AT, BT, ternary_broadcast_rebind_t<AT, BT>>) ||
        (broadcastable_to<BT, result_or_decayed_t<AT>> &&
            broadcastable_to<CT, result_or_decayed_t<AT>> &&
            cpo_invocable<D, AT, result_or_decayed_t<AT>,
                result_or_decayed_t<AT>>) ||
        (broadcastable_to<AT, result_or_decayed_t<BT>> &&
            broadcastable_to<CT, result_or_decayed_t<BT>> &&
            cpo_invocable<D, result_or_decayed_t<BT>, BT,
                result_or_decayed_t<BT>>) ||
        (broadcastable_to<AT, result_or_decayed_t<CT>> &&
            broadcastable_to<BT, result_or_decayed_t<CT>> &&
            cpo_invocable<D, result_or_decayed_t<CT>, result_or_decayed_t<CT>,
                CT>)
             DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
             static constexpr auto operator()(
                 AT&& aval, BT&& bval, CT&& cval) noexcept {
        if constexpr (ternary_broadcastable<AT, BT, CT>) {
            using T = ternary_broadcast_rebind_t<BT, CT>;
            return D::operator()(fwd::broadcast<T>(__DPL forward<AT>(aval)),
                __DPL forward<BT>(bval), __DPL forward<CT>(cval));
        } else if constexpr (ternary_broadcastable<BT, AT, CT>) {
            using T = ternary_broadcast_rebind_t<AT, CT>;
            return D::operator()(__DPL forward<AT>(aval),
                fwd::broadcast<T>(__DPL forward<BT>(bval)),
                __DPL forward<CT>(cval));
        } else if constexpr (ternary_broadcastable<CT, AT, BT>) {
            using T = ternary_broadcast_rebind_t<AT, BT>;
            return D::operator()(__DPL forward<AT>(aval),
                __DPL forward<BT>(bval),
                fwd::broadcast<T>(__DPL forward<CT>(cval)));
        } else if constexpr (simd_type<AT>) {
            using T = result_or_decayed_t<AT>;
            return D::operator()(__DPL forward<AT>(aval),
                fwd::broadcast<T>(__DPL forward<BT>(bval)),
                fwd::broadcast<T>(__DPL forward<CT>(cval)));
        } else if constexpr (simd_type<BT>) {
            using T = result_or_decayed_t<BT>;
            return D::operator()(fwd::broadcast<T>(__DPL forward<AT>(aval)),
                __DPL forward<BT>(bval),
                fwd::broadcast<T>(__DPL forward<CT>(cval)));
        } else {
            using T = result_or_decayed_t<CT>;
            return D::operator()(fwd::broadcast<T>(__DPL forward<AT>(aval)),
                fwd::broadcast<T>(__DPL forward<BT>(bval)),
                __DPL forward<CT>(cval));
        }
    }
};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
