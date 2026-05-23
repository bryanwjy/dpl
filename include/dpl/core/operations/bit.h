// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/reinterpret.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/const_mask_like.h"
#  include "dpl/core/concepts/integral_simd.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/std/bit/byteswap.h"
#  include "dpl/std/bit/countl.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void popcount(...) noexcept = delete;
void countl_zero(...) noexcept = delete;
void countl_one(...) noexcept = delete;
void countr_zero(...) noexcept = delete;
void countr_one(...) noexcept = delete;
void byteswap(...) noexcept = delete;

template <typename T, typename U>
concept bit_result_simd_for = integral_simd<T> && common_size_simd_with<T, U>;

template <typename T>
concept unqualified_popcount = requires(T arg) {
    { popcount(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mpopcount = requires(T arg) {
    { popcount(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct popcount_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_vector<E, A>>(
            [](auto val) {
                auto const count = __DPL popcount(__DPL to_unsigned(val));
                return static_cast<E>(count);
            },
            arg);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr size_t DPL_VECTORCALL fallback(
        basic_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            if constexpr (I > 0) {
                return arg[I] + self(arg, imm<I - 1>);
            } else {
                return arg[I];
            }
        }(arg, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <integral_simd T>
    requires fixed_width_vector<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_popcount<T>) {
            if constexpr (canonical_vector<T>) {
                using E = typename decltype(popcount(
                    internal::abi<T>, arg))::value_type;
                if consteval {
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return popcount(internal::abi<T>, arg);
                }
            } else {
                return popcount(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <integral_simd T>
    requires scalable_vector<T> &&
        (unqualified_popcount<T> || unqualified_popcount<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_popcount<T>) {
            return popcount(internal::abi<T>, arg);
        } else {
            return popcount(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mpopcount<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return popcount(internal::abi<T>, arg);
                }
            } else {
                return popcount(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires scalable_mask<T> &&
        (unqualified_mpopcount<T> || unqualified_mpopcount<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mpopcount<T>) {
            return popcount(internal::abi<T>, arg);
        } else {
            return popcount(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL popcount(T::value);
    }

    template <const_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { popcount(val) } -> core_convertible_to<size_t>;
                      }) {
            return popcount(val);
        } else {
            return __DPL popcount(T::value);
        }
    }
};

template <typename T>
concept unqualified_countl_zero = requires(T arg) {
    { countl_zero(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountl_zero = requires(T arg) {
    { countl_zero(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countl_zero_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_vector<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_vector<E, A>>(arg, [](auto val) {
            auto const count = __DPL countl_zero(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t fallback(basic_mask<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = !arg[I];
            if constexpr (I > 0) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return val;
            }
        }(arg, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countl_zero<T>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(countl_zero(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countl_zero(internal::abi<T>, arg);
                }
            } else {
                return countl_zero(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <integral_simd T>
    requires scalable_vector<T> &&
        (unqualified_countl_zero<T> ||
            unqualified_countl_zero<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countl_zero<T>) {
            return countl_zero(internal::abi<T>, arg);
        } else {
            return countl_zero(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_zero<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countl_zero(internal::abi<T>, arg);
                }
            } else {
                return countl_zero(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires scalable_mask<T> &&
        (unqualified_mcountl_zero<T> ||
            unqualified_mcountl_zero<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_zero<T>) {
            return countl_zero(internal::abi<T>, arg);
        } else {
            return countl_zero(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countl_zero(T::value);
    }

    template <const_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countl_zero(val) } -> core_convertible_to<size_t>;
                      }) {
            return countl_zero(val);
        } else {
            constexpr auto mask =
                static_cast<typename T::value_type>((1ll << T::width) - 1);
            constexpr auto remainder =
                sizeof(typename T::value_type) * char_bit_v - T::width;
            return __DPL countl_zero(T::value & mask) - remainder;
        }
    }
};

template <typename T>
concept unqualified_countl_one = requires(T arg) {
    { countl_one(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountl_one = requires(T arg) {
    { countl_one(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countl_one_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_vector<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_vector<E, A>>(arg, [](auto val) {
            auto const count = __DPL countl_one(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_mask<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = arg[I];
            if constexpr (I > 0) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return val;
            }
        }(arg, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countl_one<T>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(countl_one(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countl_one(internal::abi<T>, arg);
                }
            } else {
                return countl_one(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <integral_simd T>
    requires scalable_vector<T> &&
        (unqualified_countl_one<T> ||
            unqualified_countl_one<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countl_one<T>) {
            return countl_one(internal::abi<T>, arg);
        } else {
            return countl_one(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_one<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countl_one(internal::abi<T>, arg);
                }
            } else {
                return countl_one(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires scalable_mask<T> &&
        (unqualified_mcountl_one<T> ||
            unqualified_mcountl_one<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountl_one<T>) {
            return countl_one(internal::abi<T>, arg);
        } else {
            return countl_one(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countl_one(T::value);
    }

    template <const_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countl_one(val) } -> core_convertible_to<size_t>;
                      }) {
            return countl_one(val);
        } else {
            constexpr auto mask =
                static_cast<typename T::value_type>(-1ll << T::width);
            constexpr auto remainder =
                sizeof(typename T::value_type) * char_bit_v - T::width;
            return __DPL countl_one(T::value | mask) - remainder;
        }
    }
};

template <typename T>
concept unqualified_countr_zero = requires(T arg) {
    { countr_zero(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountr_zero = requires(T arg) {
    { countr_zero(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countr_zero_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_vector<E, A> arg) noexcept {
        return internal::transform<basic_vector<E, A>>(arg, [](auto val) {
            auto const count = __DPL countr_zero(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto fallback(basic_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = !arg[I];
            if constexpr (I < simd_abi_traits<E, A>::size) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return 0;
            }
        }(arg, imm<0>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countr_zero<T>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(countr_zero(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countr_zero(internal::abi<T>, arg);
                }
            } else {
                return countr_zero(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <integral_simd T>
    requires scalable_vector<T> &&
        (unqualified_countr_zero<T> ||
            unqualified_countr_zero<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countr_zero<T>) {
            return countr_zero(internal::abi<T>, arg);
        } else {
            return countr_zero(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_zero<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countr_zero(internal::abi<T>, arg);
                }
            } else {
                return countr_zero(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires scalable_mask<T> &&
        (unqualified_mcountr_zero<T> ||
            unqualified_mcountr_zero<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_zero<T>) {
            return countr_zero(internal::abi<T>, arg);
        } else {
            return countr_zero(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countr_zero(T::value);
    }

    template <const_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countr_zero(val) } -> core_convertible_to<size_t>;
                      }) {
            return countr_zero(val);
        } else {
            auto const count = __DPL countr_zero(T::value);
            return count < T::width ? count : T::width;
        }
    }
};

template <typename T>
concept unqualified_countr_one = requires(T arg) {
    { countr_one(internal::abi<T>, arg) } -> bit_result_simd_for<T>;
};

template <typename T>
concept unqualified_mcountr_one = requires(T arg) {
    { countr_one(internal::abi<T>, arg) } -> core_convertible_to<size_t>;
};

struct countr_one_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> arg) noexcept {
        return internal::transform<basic_vector<E, A>>(arg, [](auto val) {
            auto const count = __DPL countr_one(__DPL to_unsigned(val));
            return static_cast<E>(count);
        });
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> arg) noexcept {
        return []<size_t I>(this auto self, auto arg, immediate<I>) {
            auto val = arg[I];
            if constexpr (I < simd_abi_traits<E, A>::size) {
                auto const mask = val ? -1 : 0;
                return val + (mask & self(arg, imm<I - 1>));
            } else {
                return 0;
            }
        }(arg, imm<0>);
    }

public:
    template <integral_simd T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_countr_one<T>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(countr_one(
                        internal::abi<T>, arg))::value_type;
                    return dx::reinterpret<E>(fallback(arg));
                } else {
                    return countr_one(internal::abi<T>, arg);
                }
            } else {
                return countr_one(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <integral_simd T>
    requires scalable_vector<T> &&
        (unqualified_countr_one<T> ||
            unqualified_countr_one<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_countr_one<T>) {
            return countr_one(internal::abi<T>, arg);
        } else {
            return countr_one(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires fixed_width_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_one<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return countr_one(internal::abi<T>, arg);
                }
            } else {
                return countr_one(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <simd_mask T>
    requires scalable_mask<T> &&
        (unqualified_mcountr_one<T> ||
            unqualified_mcountr_one<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T arg) noexcept {
        if constexpr (unqualified_mcountr_one<T>) {
            return countr_one(internal::abi<T>, arg);
        } else {
            return countr_one(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <integral_constant_like T>
    requires unsigned_integral<typename T::value_type>
    static consteval size_t operator()(T) noexcept {
        return __DPL countr_one(T::value);
    }

    template <const_mask_like T>
    static consteval size_t operator()(T val) noexcept {
        if constexpr (requires {
                          { countr_one(val) } -> core_convertible_to<size_t>;
                      }) {
            return countr_one(val);
        } else {
            auto const count = __DPL countr_one(T::value);
            return count < T::width ? count : T::width;
        }
    }
};

template <typename T>
concept unqualified_byteswap = requires(T arg) {
    { byteswap(internal::abi<T>, arg) } -> equivalent_simd_as<T>;
};

struct byteswap_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> arg) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return internal::transform<basic_vector<E, A>>(
            arg, [](auto val) { return __DPL byteswap(val); });
    }

public:
    template <integral_simd T>
    requires fixed_width_vector<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_byteswap<T>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return byteswap(internal::abi<T>, arg);
                }
            } else {
                return byteswap(internal::abi<T>, arg);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }

    template <integral_simd T>
    requires scalable_vector<T> &&
        (unqualified_byteswap<T> || unqualified_byteswap<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_byteswap<T>) {
            return byteswap(internal::abi<T>, arg);
        } else {
            return byteswap(internal::abi<T>, dx::to_canonical(arg));
        }
    }

    template <integral_constant_like T>
    static consteval auto operator()(T) noexcept {
        return __DPL byteswap(T::value);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::popcount_t popcount{};
DPL_EXPORT inline constexpr internal::countl_zero_t countl_zero{};
DPL_EXPORT inline constexpr internal::countl_one_t countl_one{};
DPL_EXPORT inline constexpr internal::countr_zero_t countr_zero{};
DPL_EXPORT inline constexpr internal::countr_one_t countr_one{};
DPL_EXPORT inline constexpr internal::byteswap_t byteswap{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
