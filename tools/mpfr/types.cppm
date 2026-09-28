// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module mpfr:types;
import :utils;
import dpl;

export namespace mpfr {
template <size_t N>
struct type_name : static_string<N> {
private:
    using static_string<N>::data;

public:
    using static_string<N>::static_string;
};

template <size_t N>
type_name(char const (&data)[N]) -> type_name<N - 1>;

template <uint64 V>
struct type_hash : dpl::integral_constant<uint64, V> {

    using dpl::integral_constant<uint64, V>::integral_constant;
    friend consteval auto to_type(type_hash) noexcept;
};

template <typename T>
struct type_tag : details::expectation_tag<T> {
    friend consteval decltype(auto) to_type_name(type_tag) noexcept;
};

template <type_name name, typename T>
struct type_map {
    using name_type = dpl::decay_t<decltype(name)>;
    using hash_type = type_hash<string_hash(name)>;
    friend consteval auto to_type(hash_type) noexcept { return type_tag<T>{}; }

    friend consteval decltype(auto) to_type_name(type_tag<T>) noexcept {
        return (name);
    }
};

template <typename T>
requires requires(type_tag<T> tag) { to_type_name(tag); }
consteval decltype(auto) to_type_name(dpl::type_identity<T> = {}) noexcept {
    constexpr type_tag<T> tag;
    return to_type_name(tag);
}

template <uint64 V>
requires requires(type_hash<V> hash) { to_type(hash); }
consteval auto to_type(dpl::integral_constant<uint64, V> = {}) noexcept {
    constexpr type_hash<V> hash;
    return to_type(hash);
}

inline namespace literals {
template <type_name S>
consteval decltype(auto) operator""_tp_name() noexcept {
    return (S);
}
} // namespace literals

#define DEFINE_FLOAT_TYPE(NAME, TP)                              \
    [] {                                                         \
        constexpr auto name = DPL_CONCAT(#NAME, _tp_name);       \
        static_assert(sizeof(mpfr::expectation_type::NAME) > 0); \
        static_assert(sizeof(mpfr::type_map<name, TP>) > 0);     \
        return name;                                             \
    }()

inline constexpr dpl::constant_type_pack<        //
    DEFINE_FLOAT_TYPE(f16, dpl::ext::float16),   //
    DEFINE_FLOAT_TYPE(bf16, dpl::ext::bfloat16), //
    DEFINE_FLOAT_TYPE(f32, float),               //
    DEFINE_FLOAT_TYPE(f64, double)>
    float_types = {};

#undef DEFINE_FLOAT_TYPE

} // namespace mpfr
