// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include "mpfr.h"

export module mpfr:functions;
import :utils;
import dpl;

export namespace mpfr {
template <size_t N>
struct function_name : static_string<N> {
private:
    using static_string<N>::data;

public:
    using static_string<N>::static_string;
};

template <size_t N>
function_name(char const (&data)[N]) -> function_name<N - 1>;

template <uint64 V>
struct function_hash : dpl::integral_constant<uint64, V> {
    using dpl::integral_constant<uint64, V>::integral_constant;
    friend consteval auto to_function(function_hash opt) noexcept;

    friend auto invoke_from_hash(function_hash, mpfr_ptr) noexcept;
    friend auto invoke_from_hash(function_hash, mpfr_ptr, mpfr_srcptr) noexcept;
    friend auto invoke_from_hash(
        function_hash, mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t) noexcept;
    friend auto invoke_from_hash(
        function_hash, mpfr_ptr, mpfr_srcptr, mpfr_rnd_t) noexcept;
    friend auto invoke_from_hash(
        function_hash, mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t) noexcept;
};

template <auto name, typename F, F func>
struct function_map_impl {};

template <auto name, typename R, typename... Args, R (*func)(Args...)>
struct function_map_impl<name, R (*)(Args...), func> {
    using hash_type = function_hash<string_hash(name)>;

    friend auto invoke_from_hash(hash_type, Args... args) noexcept {
        if constexpr (dpl::is_same_v<R, void>) {
            return 0;
        } else {
            return func(args...);
        }
    }
};

template <typename T>
struct remove_ptr {
    using type = T;
};
template <typename T>
struct remove_ptr<T*> {
    using type = T;
};

template <typename T>
using remove_ptr_t = typename remove_ptr<T>::type;

template <auto name, auto func>
struct function_map :
    function_map_impl<name, dpl::decay_t<decltype(func)>, func> {
    using name_type = dpl::decay_t<decltype(name)>;
    using hash_type = function_hash<string_hash(name)>;

    friend consteval auto to_function(hash_type opt) noexcept { return name; }
};

template <typename T>
constexpr bool is_function_name_v = false;
template <size_t N>
constexpr bool is_function_name_v<function_name<N>> = true;
template <typename T>
concept function_name_type = is_function_name_v<T>;

inline namespace literals {
template <function_name S>
consteval auto operator""_func() noexcept {
    return S;
}
} // namespace literals

template <function_name_type auto name>
struct invoke_t {
    using hash_type = function_hash<string_hash(name)>;

    template <typename... Args>
    static auto operator()(Args... args) noexcept
    // requires requires(hash_type hash) { invoke_from_hash(hash, args...); }
    {
        constexpr hash_type hash;
        return invoke_from_hash(hash, args...);
    }
};

template <function_name_type auto name>
inline constexpr invoke_t<name> invoke;

} // namespace mpfr
