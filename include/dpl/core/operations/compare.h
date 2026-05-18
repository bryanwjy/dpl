// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_order_type.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/core/type_traits/make_simd_mask_type.h"
#  include "dpl/std/type_traits/is_invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpeq(...) noexcept = delete;
void cmpneq(...) noexcept = delete;
void cmplt(...) noexcept = delete;
void cmple(...) noexcept = delete;
void cmpgt(...) noexcept = delete;
void cmpge(...) noexcept = delete;

template <typename T, typename L, typename R>
concept compare_result =
    common_order_simd_with<T, make_simd_mask_type_t<common_order_simd_t<L, R>>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmpeq = requires(L lhs, R rhs) {
    { cmpeq(internal::abi<A>, lhs, rhs) } -> compare_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmpneq = requires(L lhs, R rhs) {
    { cmpneq(internal::abi<A>, lhs, rhs) } -> compare_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmplt = requires(L lhs, R rhs) {
    { cmplt(internal::abi<A>, lhs, rhs) } -> compare_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmple = requires(L lhs, R rhs) {
    { cmple(internal::abi<A>, lhs, rhs) } -> compare_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmpgt = requires(L lhs, R rhs) {
    { cmpgt(internal::abi<A>, lhs, rhs) } -> compare_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmpge = requires(L lhs, R rhs) {
    { cmpge(internal::abi<A>, lhs, rhs) } -> compare_result<L, R>;
};

template <typename T, typename L, typename R>
concept compare_mask_result =
    common_size_simd_with<T, common_size_simd_t<L, R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mcmpeq = requires(L lhs, R rhs) {
    { cmpeq(internal::abi<A>, lhs, rhs) } -> compare_mask_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mcmpneq = requires(L lhs, R rhs) {
    { cmpneq(internal::abi<A>, lhs, rhs) } -> compare_mask_result<L, R>;
};

struct cmpeq_t : binary_operation_base<cmpeq_t> {
private:
    friend binary_operation_base<cmpeq_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { cmpeq(internal::abi<A>, left, right) } -> simd_with_abi<A>;
    }
    {
        return cmpeq(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs == rhs; }, lhs, rhs);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> lhs, basic_mask<R, A> rhs) noexcept {
        using T = common_size_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs == rhs; }, lhs, rhs);
    }

public:
    template <simd_vector L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_cmpeq<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmpeq(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return cmpeq(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmpeq(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_cmpeq<L, R> ||
            unqualified_cmpeq<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_cmpeq<L, R>) {
            return cmpeq(internal::abi<A>, lhs, rhs);
        } else {
            return cmpeq(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_mask L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mcmpeq<L, R, A>) {
            if constexpr (canonical_mask<L> && canonical_mask<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmpeq(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return cmpeq(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmpeq(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_mask<L> && canonical_mask<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_mask L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mcmpeq<L, R> ||
            unqualified_mcmpeq<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mcmpeq<L, R>) {
            return cmpeq(internal::abi<A>, lhs, rhs);
        } else {
            return cmpeq(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpeq_t>::operator();
};

struct cmpneq_t : binary_operation_base<cmpneq_t> {
private:
    friend binary_operation_base<cmpneq_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_cmpneq<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return cmpneq(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs != rhs; }, lhs, rhs);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> lhs, basic_mask<R, A> rhs) noexcept {
        using T = common_size_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs != rhs; }, lhs, rhs);
    }

public:
    template <simd_vector L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_cmpneq<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmpneq(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return cmpneq(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmpneq(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_cmpneq<L, R> ||
            unqualified_cmpneq<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_cmpneq<L, R>) {
            return cmpneq(internal::abi<A>, lhs, rhs);
        } else {
            return cmpneq(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_mask L, common_size_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_mcmpneq<L, R, A>) {
            if constexpr (canonical_mask<L> && canonical_mask<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmpneq(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return cmpneq(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmpneq(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_mask<L> && canonical_mask<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_mask L, common_size_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_mcmpneq<L, R> ||
            unqualified_mcmpneq<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_mcmpneq<L, R>) {
            return cmpneq(internal::abi<A>, lhs, rhs);
        } else {
            return cmpneq(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpneq_t>::operator();
};

struct cmplt_t : binary_operation_base<cmplt_t> {
private:
    friend binary_operation_base<cmplt_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_cmplt<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return cmplt(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs < rhs; }, lhs, rhs);
    }

public:
    template <simd_vector L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_cmplt<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmplt(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return cmplt(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmplt(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_cmplt<L, R> ||
            unqualified_cmplt<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_cmplt<L, R>) {
            return cmplt(internal::abi<A>, lhs, rhs);
        } else {
            return cmplt(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmplt_t>::operator();
};

struct cmple_t : binary_operation_base<cmple_t> {
private:
    friend binary_operation_base<cmple_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_cmple<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return cmple(internal::abi<A>, left, right);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs <= rhs; }, lhs, rhs);
    }

public:
    template <simd_vector L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_cmple<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmple(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return cmple(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmple(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_cmple<L, R> ||
            unqualified_cmple<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_cmple<L, R>) {
            return cmple(internal::abi<A>, lhs, rhs);
        } else {
            return cmple(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmple_t>::operator();
};

struct cmpgt_t : binary_operation_base<cmpgt_t> {
private:
    friend binary_operation_base<cmpgt_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_cmpgt<L, R, A> || unqualified_cmplt<R, L, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        if constexpr (unqualified_cmpgt<L, R, A>) {
            return cmpgt(internal::abi<A>, left, right);
        } else {
            return cmplt(internal::abi<A>, right, left);
        }
    }

public:
    template <simd_vector L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_cmpgt<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmpgt(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(cmplt_t::operator()(rhs, lhs));
                } else {
                    return cmpgt(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmpgt(internal::abi<L>, lhs, rhs);
            }
        } else {
            return cmplt_t::operator()(
                dx::to_canonical(rhs), dx::to_canonical(lhs));
        }
    }

    template <simd_vector L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_cmpgt<L, R> ||
            unqualified_cmpgt<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_cmpgt<L, R>) {
            return cmpgt(internal::abi<A>, lhs, rhs);
        } else {
            return cmpgt(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpgt_t>::operator();
};

struct cmpge_t : binary_operation_base<cmpge_t> {
private:
    friend binary_operation_base<cmpge_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_cmpge<L, R, A> || unqualified_cmple<R, L, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        if constexpr (unqualified_cmpgt<L, R, A>) {
            return cmpge(internal::abi<A>, left, right);
        } else {
            return cmple(internal::abi<A>, right, left);
        }
    }

public:
    template <simd_vector L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_cmpge<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = simd_lane_type_t<decltype(cmpgt(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(cmple_t::operator()(rhs, lhs));
                } else {
                    return cmpge(internal::abi<L>, lhs, rhs);
                }
            } else {
                return cmpge(internal::abi<L>, lhs, rhs);
            }
        } else {
            return cmple_t::operator()(
                dx::to_canonical(rhs), dx::to_canonical(lhs));
        }
    }

    template <simd_vector L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_cmpge<L, R> ||
            unqualified_cmpge<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_cmpge<L, R>) {
            return cmpge(internal::abi<A>, lhs, rhs);
        } else {
            return cmpge(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpge_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cmpeq_t cmpeq{};
DPL_EXPORT inline constexpr internal::cmpneq_t cmpneq{};
DPL_EXPORT inline constexpr internal::cmplt_t cmplt{};
DPL_EXPORT inline constexpr internal::cmple_t cmple{};
DPL_EXPORT inline constexpr internal::cmpgt_t cmpgt{};
DPL_EXPORT inline constexpr internal::cmpge_t cmpge{};
} // namespace cpo

DPL_EXPORT template <typename D>
class comparison_simd_interface {
public:
    template <typename R>
    requires regular_invocable<internal::cmpeq_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpeq_t, D, R> operator==(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpeq(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmpneq_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpneq_t, D, R> operator!=(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpneq(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmplt_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmplt_t, D, R> operator<(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmplt(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmple_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmple_t, D, R> operator<=(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmple(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmpgt_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpgt_t, D, R> operator>(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpgt(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::cmpge_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::cmpge_t, D, R> operator>=(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpge(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpeq_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpeq_t, L, D> operator==(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpeq(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpneq_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpneq_t, L, D> operator!=(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpneq(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmplt_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmplt_t, L, D> operator<(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmplt(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmple_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmple_t, L, D> operator<=(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmple(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpgt_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpgt_t, L, D> operator>(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpgt(lhs, rhs);
    }

    template <typename L>
    requires regular_invocable<internal::cmpge_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::cmpge_t, L, D> operator>=(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::cmpge(lhs, rhs);
    }
};

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpeq_t, L, R> operator==(
    L lhs, R rhs) noexcept {
    return datapar::cmpeq(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpneq_t, L, R> operator!=(
    L lhs, R rhs) noexcept {
    return datapar::cmpneq(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmplt_t, L, R> operator<(
    L lhs, R rhs) noexcept {
    return datapar::cmplt(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmple_t, L, R> operator<=(
    L lhs, R rhs) noexcept {
    return datapar::cmple(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpgt_t, L, R> operator>(
    L lhs, R rhs) noexcept {
    return datapar::cmpgt(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::cmpge_t, L, R> operator>=(
    L lhs, R rhs) noexcept {
    return datapar::cmpge(lhs, rhs);
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
