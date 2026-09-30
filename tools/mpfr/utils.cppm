// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <stdlib.h>
#include <string.h>

#include <string_view>

export module mpfr:utils;
export import mpfr.interface;
import :jump_table;
import :mapped_memory;
import dpl;

export namespace mpfr {

using dpl::size_t;
using dpl::uint32;
using dpl::uint64;

constexpr uint64 string_hash(std::string_view data) noexcept {
    constexpr auto prime = 0x1000193u;
    constexpr auto offset = 0x811C9DC5u;

    auto hash = offset;

    for (auto str = data; !str.empty();
        str.remove_prefix(std::min(sizeof(uint32), str.size()))) {
        alignas(uint32) char chunk[sizeof(uint32)] = {};
        str.substr(0zu, std::min(sizeof(uint32), str.size()))
            .copy(chunk, sizeof(chunk));
        hash ^= std::bit_cast<uint32>(chunk);
        hash *= prime;
    }

    return static_cast<uint64>(data.size() << 32) | static_cast<uint64>(hash);
}

template <size_t N>
struct static_string {
    static_assert(N <= 16);
    char data[N + 1];

    consteval std::string_view operator+() const noexcept {
        return static_cast<std::string_view>(*this);
    }

    consteval char const* c_str() const noexcept { return data; }
    consteval size_t size() const noexcept { return N; }

    consteval operator std::string_view() const noexcept {
        return std::string_view(data, N);
    }
    consteval static_string(char const (&str)[N + 1]) noexcept
        : static_string(str, dpl::make_index_sequence<N + 1>{}) {}

    template <size_t... Is>
    requires (sizeof...(Is) == N + 1)
    consteval static_string(
        char const (&str)[N + 1], dpl::index_sequence<Is...>) noexcept
        : data{str[Is]...} {}
};

class tmp_string {
public:
    constexpr tmp_string(std::string_view from) noexcept : buffer_{} {
        if (from.size() >= sizeof(buffer_)) {
            ptr_ = strndup(from.data(), from.size());
        } else {
            from.copy(buffer_, sizeof(buffer_) - 1);
            ptr_ = buffer_;
        }
    }

    explicit constexpr tmp_string(size_t length) noexcept : buffer_{} {
        if (length >= sizeof(buffer_)) {
            ptr_ = (char*)calloc(length + 1, sizeof(char));
        } else {
            ptr_ = buffer_;
        }
    }

    constexpr char const* c_str() const noexcept { return ptr_; }
    constexpr char const* data() const noexcept { return ptr_; }
    constexpr char* data() noexcept { return ptr_; }
    constexpr operator std::string_view() const noexcept {
        return std::string_view(data());
    }

    tmp_string(tmp_string const&) = delete;
    tmp_string& operator=(tmp_string const&) = delete;
    tmp_string(tmp_string&& other) noexcept {
        if (other.ptr_ != other.buffer_) {
            ptr_ = dpl::exchange(other.ptr_, nullptr);
        } else {
            strcpy(buffer_, other.buffer_);
            ptr_ = buffer_;
        }
    }
    tmp_string& operator=(tmp_string&& other) noexcept {
        if (other.ptr_ != other.buffer_) {
            ptr_ = dpl::exchange(other.ptr_, nullptr);
        } else {
            strcpy(buffer_, other.buffer_);
            ptr_ = buffer_;
        }
        return *this;
    }

    constexpr ~tmp_string() {
        if (ptr_ != buffer_) {
            free(ptr_);
        }
    }

private:
    char buffer_[120];
    char* ptr_;
};

constexpr std::string_view trim(std::string_view view) noexcept {
    constexpr std::string_view space = " \n\t\r";
    auto const first = view.find_first_not_of(space);
    if (first == std::string_view::npos)
        return view.substr(
            view.size()); // empty, but still points into the input
    auto const last = view.find_last_not_of(space);
    return view.substr(first, last - first + 1);
}

constexpr std::string_view strip(std::string_view view) noexcept {
    view = trim(view);
    if (view.size() >= 2 && (view.front() == '\'' || view.front() == '"') &&
        view.back() == view.front())
        view = trim(view.substr(1, view.size() - 2));
    return view;
}

template <typename T>
struct rebind_as_type_pack {};
template <template <typename...> class C, typename... Ts>
struct rebind_as_type_pack<C<Ts...>> {
    using type = dpl::type_pack<Ts...>;
};
template <typename T>
using rebind_as_type_pack_t = typename rebind_as_type_pack<T>::type;

template <typename T>
requires requires { typename rebind_as_type_pack_t<T>; }
consteval auto to_type_pack() noexcept {
    return rebind_as_type_pack_t<T>{};
}

template <typename T>
struct unwrap_type : dpl::type_identity<T> {};
template <typename T>
using unwrap_type_t = typename unwrap_type<T>::type;

template <typename T>
requires requires { typename T::type; }
struct unwrap_type<T> : dpl::type_identity<typename T::type> {};

template <typename T>
struct remove_ptr : dpl::type_identity<T> {};
template <typename T>
using remove_ptr_t = typename remove_ptr<T>::type;
template <typename T>
struct remove_ptr<T*> : dpl::type_identity<T> {};

template <typename T>
struct last_element {};
template <typename T>
using last_element_t = typename last_element<T>::type;

template <typename T>
requires requires { typename rebind_as_type_pack_t<T>; }
struct last_element<T> : last_element<rebind_as_type_pack_t<T>> {};
template <typename... Ts>
struct last_element<dpl::type_pack<Ts...>> :
    remove_ptr<decltype((..., static_cast<Ts*>(0)))> {};

template <typename T>
struct first_element {};
template <typename T>
using first_element_t = typename first_element<T>::type;

template <typename T>
requires requires { typename rebind_as_type_pack_t<T>; }
struct first_element<T> : first_element<rebind_as_type_pack_t<T>> {};
template <typename T, typename... Ts>
struct first_element<dpl::type_pack<T, Ts...>> : dpl::type_identity<T> {};

} // namespace mpfr
