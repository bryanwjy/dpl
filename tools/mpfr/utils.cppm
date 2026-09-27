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

    constexpr char const* c_str() const noexcept { return ptr_; }
    constexpr char const* data() const noexcept { return ptr_; }
    constexpr char* data() noexcept { return ptr_; }
    tmp_string(tmp_string const&) = delete;
    tmp_string& operator=(tmp_string const&) = delete;
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

} // namespace mpfr
