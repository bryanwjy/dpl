// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/negate.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void fmadd(...) noexcept = delete;
void fmsub(...) noexcept = delete;
void fnmadd(...) noexcept = delete;
void fnmsub(...) noexcept = delete;
void fmsubadd(...) noexcept = delete;
void fmaddsub(...) noexcept = delete;

template <typename T, typename L, typename M, typename R>
concept fma_result = common_float_simd_with<common_float_simd_t<L, M, R>, T>;

template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_fmadd = requires(L a, M b, R c) {
    { fmadd(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};
template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_fmsub = requires(L a, M b, R c) {
    { fmsub(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};

template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_fnmadd = requires(L a, M b, R c) {
    { fnmadd(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};
template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_fnmsub = requires(L a, M b, R c) {
    { fnmsub(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};

template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_fmaddsub = requires(L a, M b, R c) {
    { fmaddsub(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};
template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_fmsubadd = requires(L a, M b, R c) {
    { fmsubadd(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};

template <typename A, typename B, typename C>
concept only_unqualified_ternary = !canonical_vector<A> ||
    !canonical_vector<B> || !canonical_vector<C> || !same_abi_simd_as<A, B> ||
    !same_abi_simd_as<A, C> || !same_abi_simd_as<B, C>;

template <typename T>
struct fma_base {
    template <floating_point_simd TA, broadcastable_to<TA> TB,
        broadcastable_to<TA> TC>
    requires regular_invocable<T, TA, TA, TA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TA, TA, TB, TC>) {
            if constexpr (canonical_vector<TA>) {
                if consteval {
                    return T::operator()(
                        a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
                } else {
                    return T::native(internal::abi<TA>, a, b, c);
                }
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA>) {
            return T::operator()(a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
        } else {
            return T::operator()(dx::to_canonical(a), b, c);
        }
    }

    template <floating_point_simd TB, broadcastable_to<TB> TA,
        broadcastable_to<TB> TC>
    requires regular_invocable<T, TB, TB, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TB, TA, TB, TC>) {
            if constexpr (canonical_vector<TB>) {
                if consteval {
                    return T::operator()(
                        dx::broadcast<TB>(a), b, dx::broadcast<TB>(c));
                } else {
                    return T::native(internal::abi<TA>, a, b, c);
                }
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else if constexpr (canonical_vector<TB>) {
            return T::operator()(dx::broadcast<TB>(a), b, dx::broadcast<TB>(c));
        } else {
            return T::operator()(a, dx::to_canonical(b), c);
        }
    }

    template <floating_point_simd TC, broadcastable_to<TC> TA,
        broadcastable_to<TC> TB>
    requires regular_invocable<T, TC, TC, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        if constexpr (implements_native<T, TC, TA, TB, TC>) {
            if constexpr (canonical_vector<TC>) {
                if consteval {
                    return T::operator()(
                        dx::broadcast<TC>(a), dx::broadcast<TC>(b), c);
                } else {
                    return T::native(internal::abi<TA>, a, b, c);
                }
            } else {
                return T::native(internal::abi<TA>, a, b, c);
            }
        } else if constexpr (canonical_vector<TC>) {
            return T::operator()(dx::broadcast<TC>(a), dx::broadcast<TC>(b), c);
        } else {
            return T::operator()(a, b, dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_arithmetic_simd_with<TA> TB,
        broadcastable_to<common_arithmetic_simd_t<TA, TB>> TC>
    requires regular_invocable<T, TA, TB, common_arithmetic_simd_t<TA, TB>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB>;
        if constexpr (implements_native<T, A, TA, TB, TC>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB>) {
                if consteval {
                    return T::operator()(a, b,
                        dx::broadcast<common_arithmetic_simd_t<TA, TB>>(c));
                } else {
                    return T::native(internal::abi<A>, a, b, c);
                }
            } else {
                return T::native(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB>) {
            return T::operator()(
                a, b, dx::broadcast<common_arithmetic_simd_t<TA, TB>>(c));
        } else {
            return T::operator()(dx::to_canonical(a), dx::to_canonical(b), c);
        }
    }

    template <floating_point_simd TA, common_arithmetic_simd_with<TA> TC,
        broadcastable_to<common_arithmetic_simd_t<TA, TC>> TB>
    requires regular_invocable<T, TA, common_arithmetic_simd_t<TA, TC>, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TC>;
        if constexpr (implements_native<T, A, TA, TB, TC>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TC>) {
                if consteval {
                    return T::operator()(a,
                        dx::broadcast<common_arithmetic_simd_t<TA, TC>>(b), c);
                } else {
                    return T::native(internal::abi<A>, a, b, c);
                }
            } else {
                return T::native(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TC>) {
            return T::operator()(
                a, dx::broadcast<common_arithmetic_simd_t<TA, TC>>(b), c);
        } else {
            return T::operator()(dx::to_canonical(a), b, dx::to_canonical(c));
        }
    }

    template <floating_point_simd TB, common_arithmetic_simd_with<TB> TC,
        broadcastable_to<common_arithmetic_simd_t<TB, TC>> TA>
    requires regular_invocable<T, common_arithmetic_simd_t<TB, TC>, TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TB, TC>;
        if constexpr (implements_native<T, A, TA, TB, TC>) {
            if constexpr (canonical_vector<TB> && canonical_vector<TC>) {
                if consteval {
                    return T::operator()(
                        dx::broadcast<common_arithmetic_simd_t<TB, TC>>(a), b,
                        c);
                } else {
                    return T::native(internal::abi<A>, a, b, c);
                }
            } else {
                return T::native(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TB> && canonical_vector<TC>) {
            return T::operator()(
                dx::broadcast<common_arithmetic_simd_t<TB, TC>>(a), b, c);
        } else {
            return T::operator()(a, dx::to_canonical(b), dx::to_canonical(c));
        }
    }
};

struct fmadd_t : fma_base<fmadd_t> {
private:
    friend fma_base<fmadd_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA a, TB b, TC c) noexcept
    requires requires {
        { fmadd(abi, a, b, c) } -> floating_point_simd_with_abi<A>;
    }
    {
        return fmadd(abi, a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD) static constexpr auto DPL_VECTORCALL
    fallback(basic_vector<E, A> a, basic_vector<E, A> b,
        basic_vector<E, A> c) noexcept {
        return dx::add(dx::multiply(a, b), c);
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_fmadd<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return fmadd(internal::abi<A>, a, b, c);
                }
            } else {
                return fmadd(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_fmadd<TA, TB, TC> ||
            unqualified_fmadd<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_fmadd<TA, TB, TC>) {
            return fmadd(internal::abi<A>, a, b, c);
        } else {
            return fmadd(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<fmadd_t>::operator();
};

struct fmsub_t : fma_base<fmsub_t> {
private:
    friend fma_base<fmsub_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA a, TB b, TC c) noexcept
    requires requires {
        { fmsub(abi, a, b, c) } -> floating_point_simd_with_abi<A>;
    }
    {
        return fmsub(abi, a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return dx::subtract(dx::multiply(a, b), c);
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_fmsub<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return fmsub(internal::abi<A>, a, b, c);
                }
            } else {
                return fmsub(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_fmsub<TA, TB, TC> ||
            unqualified_fmsub<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_fmsub<TA, TB, TC>) {
            return fmsub(internal::abi<A>, a, b, c);
        } else {
            return fmsub(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<fmsub_t>::operator();
};

struct fnmadd_t : fma_base<fnmadd_t> {
private:
    friend fma_base<fnmadd_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA a, TB b, TC c) noexcept
    requires requires {
        { fnmadd(abi, a, b, c) } -> floating_point_simd_with_abi<A>;
    }
    {
        return fnmadd(abi, a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return dx::subtract(c, dx::multiply(a, b));
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_fnmadd<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return fnmadd(internal::abi<A>, a, b, c);
                }
            } else {
                return fnmadd(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_fnmadd<TA, TB, TC> ||
            unqualified_fnmadd<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_fnmadd<TA, TB, TC>) {
            return fnmadd(internal::abi<A>, a, b, c);
        } else {
            return fnmadd(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<fnmadd_t>::operator();
};

struct fnmsub_t : fma_base<fnmsub_t> {
private:
    friend fma_base<fnmsub_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA a, TB b, TC c) noexcept
    requires requires {
        { fnmsub(abi, a, b, c) } -> floating_point_simd_with_abi<A>;
    }
    {
        return fnmsub(abi, a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return dx::subtract(dx::negate(dx::multiply(a, b)), c);
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_fnmsub<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return fnmsub(internal::abi<A>, a, b, c);
                }
            } else {
                return fnmsub(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_fnmsub<TA, TB, TC> ||
            unqualified_fnmsub<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_fnmsub<TA, TB, TC>) {
            return fnmsub(internal::abi<A>, a, b, c);
        } else {
            return fnmsub(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<fnmsub_t>::operator();
};

struct fmaddsub_t : fma_base<fmaddsub_t> {
private:
    friend fma_base<fmaddsub_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA a, TB b, TC c) noexcept
    requires requires {
        { fmaddsub(abi, a, b, c) } -> floating_point_simd_with_abi<A>;
    }
    {
        return fmaddsub(abi, a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return fmadd_t::operator()(a, b, dx::negate(c, imm<0b1010>, c));
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_fmaddsub<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return fmaddsub(internal::abi<A>, a, b, c);
                }
            } else {
                return fmaddsub(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_fmaddsub<TA, TB, TC> ||
            unqualified_fmaddsub<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_fmaddsub<TA, TB, TC>) {
            return fmaddsub(internal::abi<A>, a, b, c);
        } else {
            return fmaddsub(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<fmaddsub_t>::operator();
};

struct fmsubadd_t : fma_base<fmsubadd_t> {
private:
    friend fma_base<fmsubadd_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA a, TB b, TC c) noexcept
    requires requires {
        { fmaddsub(abi, a, b, c) } -> floating_point_simd_with_abi<A>;
    }
    {
        return fmsubadd(abi, a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        return fmadd_t::operator()(a, b, dx::negate(c, imm<0b0101>, c));
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_fmsubadd<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return fmsubadd(internal::abi<A>, a, b, c);
                }
            } else {
                return fmsubadd(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_fmsubadd<TA, TB, TC> ||
            unqualified_fmsubadd<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_fmsubadd<TA, TB, TC>) {
            return fmsubadd(internal::abi<A>, a, b, c);
        } else {
            return fmsubadd(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<fmsubadd_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::fmadd_t fmadd{};
DPL_EXPORT inline constexpr internal::fmsub_t fmsub{};
DPL_EXPORT inline constexpr internal::fnmadd_t fnmadd{};
DPL_EXPORT inline constexpr internal::fnmsub_t fnmsub{};
DPL_EXPORT inline constexpr internal::fmaddsub_t fmaddsub{};
DPL_EXPORT inline constexpr internal::fmsubadd_t fmsubadd{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
