// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <algorithm>
#include <charconv>
#include <expected>
#include <ranges>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

export module mpfr:typeconv;
import :utils;
import dpl;

export namespace mpfr {

template <dpl::integral T>
inline std::expected<T, std::errc> parse(std::string_view str) noexcept {
    auto const negative = str.starts_with('-');
    constexpr auto success = std::errc(0);
    auto result = std::errc::invalid_argument;
    auto const error_path = [&] { return std::unexpected(result); };
    str.remove_prefix(negative);

    T val = 0;
    auto const success_path = [&] { return negative ? -val : val; };
    if (!str.starts_with('0') || str.size() <= 1) {
        result =
            std::from_chars(str.data(), str.data() + str.size(), val, 10).ec;
        if (result == success) {
            return success_path();
        }

        return error_path();
    }

    str.remove_prefix(1zu);
    if (str.starts_with('x') || str.starts_with('X')) {
        str.remove_prefix(1zu);
        result =
            std::from_chars(str.data(), str.data() + str.size(), val, 16).ec;
        if (result == success) {
            return success_path();
        }

        return error_path();
    }

    if (!str.starts_with('.')) {
        result =
            std::from_chars(str.data(), str.data() + str.size(), val, 8).ec;
        if (result == success) {
            return success_path();
        }

        return error_path();
    }

    // starts with "0."
    return 0;
}

template <dpl::floating_point_like T>
inline std::expected<T, std::errc> parse(std::string_view str) noexcept {
    if constexpr (sizeof(T) < sizeof(float)) {
        return parse<float>(str).transform(
            [](float val) { return static_cast<T>(val); });
    } else {
        auto const negative = str.starts_with('-');
        str.remove_prefix(negative);
        constexpr auto lowercase = [](char val) { return ::tolower(val); };

        T val = 0;
        using std::string_view_literals::operator""sv;
        if ([&] {
                // workaround compiler bug when compiling with libstdc++
                constexpr char nan[] = "nan";
                constexpr auto len = sizeof(nan) - 1;
                return std::ranges::equal(str.begin(), str.end(), nan,
                    nan + len, std::equal_to{}, lowercase);
            }()) {
            constexpr auto nanbits = ~dpl::bitset<dpl::type_bit_v<T>>();
            val = dpl::bit_cast<T>(nanbits >> !negative);
            return val;
        }

        if ([&] {
                constexpr char inf[] = "infinity";
                constexpr auto len = sizeof(inf) - 1;
                return std::ranges::equal(str.begin(), str.end(), inf,
                    inf + len, std::equal_to{}, lowercase);
            }()) {
            val =
                dpl::bit_cast<T>(dpl::floating_point_traits<T>::exponent_mask);
            return negative ? -val : val;
        }

        auto const ishex = str.starts_with("0x") || str.starts_with("0X");
        auto const format =
            ishex ? std::chars_format::hex : std::chars_format::general;
        str.remove_prefix(ishex ? 2zu : 0zu);

        auto const result =
            std::from_chars(str.data(), str.data() + str.size(), val, format)
                .ec;
        if (result != std::errc(0)) {
            return std::unexpected(result);
        }

        return negative ? -val : val;
    }
}

namespace typeconv {

struct string {};
template <typename T>
struct list;
template <typename T>
struct unbounded;
template <typename...>
struct tuple;
template <typename T>
concept scalar_argument = dpl::same_as<string, T> ||
    (dpl::is_arithmetic_v<T> && !dpl::same_as<T, bool>);

template <typename T>
inline constexpr bool is_list_v = false;
template <typename T>
inline constexpr bool is_unbounded_v = false;
template <typename T>
inline constexpr bool is_tuple_v = false;
template <typename T>
inline constexpr bool is_list_v<list<T>> = scalar_argument<T> || is_tuple_v<T>;
template <scalar_argument T>
inline constexpr bool is_unbounded_v<unbounded<T>> = true;
template <typename... Ts>
inline constexpr bool is_tuple_v<tuple<Ts...>> = (... && scalar_argument<Ts>) ||
    (is_unbounded_v<last_element_t<tuple<Ts...>>> &&
        (... + scalar_argument<Ts>) == sizeof...(Ts) - 1);

template <typename T>
using unwrap_list_t = dpl::conditional_t<is_list_v<T>, unwrap_type_t<T>, T>;
template <typename T>
using unwrap_unbounded_t =
    dpl::conditional_t<is_unbounded_v<T>, unwrap_type_t<T>, T>;

template <typename T>
concept list_element_type = scalar_argument<T> || is_tuple_v<T>;

template <list_element_type T>
struct list<T> : dpl::type_identity<T> {};

template <scalar_argument T>
struct unbounded<T> : dpl::type_identity<T> {};

template <typename... Ts>
requires is_tuple_v<tuple<Ts...>>
struct tuple<Ts...> : dpl::type_pack<Ts...> {};

template <typename T>
inline constexpr bool is_homogeneous_tuple_v = false;

template <typename T, typename... Ts>
inline constexpr bool is_homogeneous_tuple_v<tuple<T, Ts...>> =
    (... && dpl::same_as<T, unwrap_unbounded_t<Ts>>);

template <typename T>
concept list_type = is_list_v<T>;
template <typename T>
concept tuple_type = is_tuple_v<T>;
template <typename T>
concept unbounded_tuple_type =
    tuple_type<T> && is_unbounded_v<last_element_t<T>>;
template <typename T>
concept homogeneous_tuple_type = tuple_type<T> && is_homogeneous_tuple_v<T>;
template <typename T>
concept homogeneous_type = homogeneous_tuple_type<T> || scalar_argument<T> ||
    (list_type<T> &&
        (homogeneous_tuple_type<unwrap_list_t<T>> || scalar_argument<T>));

template <typename T>
concept defaultable_type =
    (tuple_type<T> && !unbounded_tuple_type<T>) || scalar_argument<T>;

template <typename>
struct storage_type {};
template <typename T>
using storage_type_t = typename storage_type<T>::type;

template <typename V, typename T>
struct make_variant {};
template <typename T>
using make_variant_t =
    typename make_variant<void, rebind_as_type_pack_t<T>>::type;

template <typename T, typename... Ts>
struct make_variant<void, dpl::type_pack<T, Ts...>> :
    make_variant<std::variant<T>, dpl::type_pack<Ts...>> {};

template <typename... Ts>
struct make_variant<std::variant<Ts...>, dpl::type_pack<>> :
    dpl::type_identity<std::variant<Ts...>> {};

template <typename... Vs, typename T, typename... Ts>
requires requires(std::variant<Vs...> v) {
    std::get<storage_type_t<unwrap_unbounded_t<T>>>(v);
}
struct make_variant<std::variant<Vs...>, dpl::type_pack<T, Ts...>> :
    make_variant<std::variant<Vs...>, dpl::type_pack<Ts...>> {};

template <typename... Vs, typename T, typename... Ts>
struct make_variant<std::variant<Vs...>, dpl::type_pack<T, Ts...>> :
    make_variant<std::variant<Vs..., storage_type_t<unwrap_unbounded_t<T>>>,
        dpl::type_pack<Ts...>> {};

template <typename T>
struct make_tuple {};
template <typename T>
using make_tuple_t = typename make_tuple<T>::type;
template <typename... Ts>
struct make_tuple<tuple<Ts...>> {
    using type = std::tuple<storage_type_t<Ts>...>;
};

template <>
struct storage_type<bool> {
    using type = bool;
};

template <scalar_argument T>
struct storage_type<T> {
    using type = T;
};

template <scalar_argument T>
requires dpl::same_as<T, string>
struct storage_type<T> {
    using type = std::string_view;
};

template <unbounded_tuple_type T>
requires homogeneous_tuple_type<T>
struct storage_type<T> {
    using type =
        std::vector<storage_type_t<unwrap_unbounded_t<first_element_t<T>>>>;
};

template <unbounded_tuple_type T>
struct storage_type<T> {
    using type = std::vector<make_variant_t<T>>;
};

template <tuple_type T>
struct storage_type<T> : make_tuple<T> {};

template <list_type T>
struct storage_type<T> {
    using type = std::vector<storage_type_t<unwrap_list_t<T>>>;
};

// don't make nested vectors
template <list_type T>
requires unbounded_tuple_type<unwrap_list_t<T>>
struct storage_type<T> : storage_type<unwrap_list_t<T>> {};

template <typename T>
struct storage_extent {
    constexpr storage_extent() noexcept = default;
};

template <typename T>
inline constexpr auto minimum_extent = [] {
    if constexpr (tuple_type<T>) {
        return dpl::tuple_size_v<rebind_as_type_pack_t<T>>;
    } else if constexpr (list_type<T> && tuple_type<unwrap_list_t<T>>) {
        return dpl::tuple_size_v<rebind_as_type_pack_t<unwrap_list_t<T>>>;
    } else if constexpr (std::same_as<bool, T>) {
        return 0zu;
    } else {
        return 1zu;
    }
}();

inline constexpr auto uninitialized_extent = -1zu;

template <typename T>
requires unbounded_tuple_type<T>
struct storage_extent<T> {
    constexpr storage_extent() noexcept = default;

    constexpr bool verify_extent(size_t chunk_size) noexcept {
        extent =
            chunk_size >= minimum_extent<T> ? chunk_size : uninitialized_extent;
        return extent == chunk_size;
    }

    size_t extent = uninitialized_extent;
};

template <typename T>
requires list_type<T> && unbounded_tuple_type<unwrap_list_t<T>>
struct storage_extent<T> {
    constexpr storage_extent() noexcept = default;

    constexpr bool verify_extent(size_t chunk_size) noexcept {
        if (extent == uninitialized_extent && chunk_size >= minimum_extent<T>) {
            extent = chunk_size;
        }

        return extent == chunk_size;
    }

    size_t extent = uninitialized_extent;
};

enum class errc {
    extent_mismatch,
    parsing_error,
    uninitialized,
};

template <scalar_argument T>
std::expected<storage_type_t<T>, typeconv::errc> parse_scalar(
    std::string_view str) noexcept {
    if constexpr (dpl::is_same_v<string, T>) {
        return str;
    } else {
        return mpfr::parse<T>(str).transform_error(
            [](std::errc) { return errc::parsing_error; });
    }
}

template <tuple_type T>
requires (!unbounded_tuple_type<T>)
std::expected<storage_type_t<T>, typeconv::errc> parse_tuple(
    auto&& chunks) noexcept {
    using pack_t = rebind_as_type_pack_t<T>;
    constexpr auto size = std::tuple_size_v<pack_t>;
    std::expected<storage_type_t<T>, typeconv::errc> result(std::in_place);
    for (auto const args : chunks | std::views::adjacent<size>) {
        dpl::pack::all_of(
            [&]<size_t I>(dpl::size_constant<I>) {
                auto const str = std::get<I>(args);
                return parse_scalar<std::tuple_element_t<I, pack_t>>(str)
                    .transform([&](auto&& scalar) {
                        std::get<I>(*result) = std::move(scalar);
                    })
                    .transform_error([&](typeconv::errc code) {
                        result = std::unexpected(code);
                    })
                    .has_value();
            },
            dpl::make_index_sequence<size>{});

        // this is deliberate, there should only be 1 chunk
        return result;
    }
}

template <typename T>
class storage : storage_extent<T> {
public:
    using type = T;
    using value_type = storage_type_t<T>;

private:
    using parse_result = std::expected<value_type, typeconv::errc>;
    using update_result = std::expected<void, typeconv::errc>;
    using base_type = storage_extent<T>;

protected:
    constexpr storage() noexcept = default;

    constexpr storage() noexcept
    requires dpl::same_as<bool, T>
        : storage_(std::in_place, false) {}

    constexpr storage(storage_type_t<T> arg)
    requires defaultable_type<T>
        : storage_(std::in_place, dpl::move(arg)) {}

    constexpr size_t extent() const noexcept {
        if constexpr (unbounded_tuple_type<T> ||
            list_type<T> && unbounded_tuple_type<unwrap_list_t<T>>) {
            return base_type::extent;
        } else {
            return minimum_extent<T>;
        }
    }

    constexpr update_result update(auto&& chunks) noexcept
    requires scalar_argument<T>
    {
        if (chunks.empty()) {
            storage_ = std::unexpected(errc::extent_mismatch);
        } else {
            storage_ = typeconv::parse_scalar<T>(chunks.front());
        }

        return storage_.transform([](auto&&) {});
    }

    constexpr update_result update(auto&&) noexcept
    requires dpl::same_as<T, bool>
    {
        storage_ = true;
        return update_result(std::in_place);
    }

    constexpr update_result update(auto&& chunks) noexcept
    requires list_type<T>
    {
        if (chunks.empty()) {
            storage_ = std::unexpected(errc::extent_mismatch);
            return std::unexpected(errc::extent_mismatch);
        }

        if (!storage_) {
            // initialize
            storage_.emplace();
        }

        using V = unwrap_list_t<type>;
        if constexpr (scalar_argument<V>) {
            return parse_scalar<V>(chunks.front())
                .transform([&](storage_type_t<V>&& val) {
                    storage_->emplace_back(dpl::move(val));
                })
                .transform_error([&](typeconv::errc error) {
                    storage_ = std::unexpected(error);
                    return error;
                });
        } else {
            static_assert(is_tuple_v<V>);
            // list of bounded tuple
            return parse_tuple<V>(chunks)
                .transform([&](storage_type_t<V>&& val) {
                    storage_->emplace_back(dpl::move(val));
                })
                .transform_error([&](typeconv::errc error) {
                    storage_ = std::unexpected(error);
                    return error;
                });
        }
    }

    constexpr update_result update(auto&& chunks) noexcept
    requires list_type<T> && unbounded_tuple_type<unwrap_list_t<T>>
    {
        using element_type = typename value_type::value_type;
        if (!base_type::verify_extent(
                dpl::to_unsigned(std::ranges::distance(chunks)))) {
            storage_ = std::unexpected(errc::extent_mismatch);
            return std::unexpected(errc::extent_mismatch);
        }

        if (!storage_) {
            // initialize
            storage_.emplace().reserve(extent());
        }

        update_result result(std::in_place);
        if constexpr (!requires { std::variant_size<element_type>::value; }) {
            for (auto const str : chunks | std::views::take_while([&](auto) {
                     return result.has_value();
                 })) {
                // first unwrap the list to a tuple
                // then grab the first element
                // then unwrap a possible unbounded
                using scalar_type =
                    unwrap_unbounded_t<first_element_t<unwrap_list_t<T>>>;
                result = typeconv::parse_scalar<scalar_type>(str)
                             .transform([&](element_type&& val) {
                                 storage_->emplace_back(dpl::move(val));
                             })
                             .transform_error([&](errc error) {
                                 storage_ = std::unexpected(error);
                                 return error;
                             });
            }

            return result;
        } else {
            constexpr auto pack = to_type_pack<unwrap_list_t<type>>();
            // element_type is a variant
            dpl::ignore = dpl::pack::all_of(
                [&]<typename U>(dpl::type_identity<U>) {
                    using V = storage_type_t<unwrap_unbounded_t<U>>;
                    auto const append = [&](std::string_view str) {
                        return typeconv::parse_scalar<element_type>(str)
                            .transform([&](element_type val) {
                                storage_->emplace_back(
                                    std::in_place_type<V>, val);
                            })
                            .transform_error([&](typeconv::errc error) {
                                storage_ = std::unexpected(error);
                                return error;
                            });
                    };

                    if constexpr (is_unbounded_v<U>) {
                        for (auto const str : chunks |
                                std::views::take_while(
                                    [&](auto) { return result.has_value(); })) {
                            result = append(str);
                        }
                    } else {
                        auto const str = chunks.front();
                        chunks.advance(1);
                        result = append(str);
                    }

                    return result.has_value();
                },
                pack);

            return result;
        }
    }

    constexpr update_result update(auto&& chunks) noexcept
    requires tuple_type<T>
    {
        if (dpl::to_unsigned(std::ranges::distance(chunks)) !=
            dpl::tuple_size_v<rebind_as_type_pack_t<T>>) {
            storage_ = std::unexpected(errc::extent_mismatch);
            return std::unexpected(errc::extent_mismatch);
        }

        return parse_tuple<T>(chunks).transform(
            [&](storage_type_t<T>&& val) { storage_ = dpl::move(val); });
    }

    constexpr update_result update(auto&& chunks) noexcept
    requires unbounded_tuple_type<T>
    {
        if (!base_type::verify_extent(
                dpl::to_unsigned(std::ranges::distance(chunks)))) {
            storage_ = std::unexpected(errc::extent_mismatch);
            return std::unexpected(errc::extent_mismatch);
        }
        if (!storage_) {
            // initialize
            storage_.emplace().resize(extent());
        }

        using element_type = typename value_type::value_type;
        update_result result(std::in_place);
        if constexpr (!requires { std::variant_size<element_type>::value; }) {
            for (auto const [idx, str] :
                chunks | std::views::take_while([&](auto str) {
                    return result.has_value();
                }) | std::views::enumerate) {
                using scalar_type = unwrap_unbounded_t<first_element_t<T>>;
                result = typeconv::parse_scalar<scalar_type>(str)
                             .transform([&](element_type val) {
                                 (*storage_)[idx] = val;
                             })
                             .transform_error([&](errc error) {
                                 storage_ = std::unexpected(error);
                                 return error;
                             });
            }

            return result;
        } else {
            constexpr auto pack = to_type_pack<type>();
            auto const begin = chunks.begin();
            // element_type is a variant
            dpl::ignore = dpl::pack::all_of(
                [&]<typename U>(dpl::type_identity<U>) {
                    using V = storage_type_t<unwrap_unbounded_t<U>>;
                    auto const assign = [&](size_t idx, std::string_view str) {
                        return typeconv::parse_scalar<element_type>(str)
                            .transform([&](element_type val) {
                                (*storage_)[idx].emplace(
                                    std::in_place_type<V>, val);
                            })
                            .transform_error([&](typeconv::errc error) {
                                storage_ = std::unexpected(error);
                                return error;
                            });
                    };

                    if constexpr (is_unbounded_v<U>) {
                        for (auto const [idx, str] :
                            chunks | std::views::take_while([&](auto) {
                                return result.has_value();
                            }) | std::views::enumerate) {
                            constexpr auto offset =
                                dpl::tuple_size_v<decltype(pack)> - 1;
                            result = assign(idx + offset, str);
                        }
                    } else {
                        auto const str = chunks.front();
                        auto const idx =
                            std::ranges::distance(begin, chunks.begin());
                        chunks.advance(1);
                        result = assign(dpl::to_unsigned(idx), str);
                    }

                    return result.has_value();
                },
                pack);

            return result;
        }
    }

    template <typename S>
    constexpr decltype(auto) get(this S&& self) noexcept {
        using self_t = dpl::copy_cv_t<dpl::remove_reference_t<S>, storage>;
        return dpl::forward_like<S>(((self_t&)self).storage_);
    }

private:
    parse_result storage_ = std::unexpected(errc::uninitialized);
};

template <typename T>
concept convertible = std::same_as<T, bool> || scalar_argument<T> ||
    list_type<T> || tuple_type<T>;

} // namespace typeconv
} // namespace mpfr
