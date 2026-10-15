// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test.support:dynamic_array;
import :span;
import :array;
import dpl;

export namespace dpl::test {
inline namespace support {
template <dpl::semiregular E>
class dynamic_array {
    static_assert(dpl::is_same_v<E, dpl::decay_t<E>>);

public:
    using pointer = E*;
    using const_pointer = E const*;
    using reference = E&;
    using const_reference = E const&;
    using size_type = size_t;
    using value_type = E;
    using difference_type = ptrdiff_t;

    constexpr dynamic_array() noexcept = default;
    constexpr ~dynamic_array() noexcept { destroy(); }

    explicit constexpr dynamic_array(size_t count)
        : begin_(new E[count])
        , end_(begin_ + count) {}

    explicit constexpr dynamic_array(span<E const> data)
        : begin_(new E[data.size()])
        , end_(begin_ + data.size()) {
        for (auto* ptr = begin_; auto const& val : data) {
            *ptr++ = val;
        }
    }

    dynamic_array(dynamic_array const& other)
        : dynamic_array(static_cast<span<E const>>(other)) {}

    dynamic_array& operator=(dynamic_array const& other) {
        if (this != &other) {
            auto const _ = dpl::exchange(
                *this, dynamic_array(static_cast<span<E const>>(other)));
        }

        return *this;
    }

    constexpr dynamic_array(dynamic_array&& other) noexcept
        : begin_(dpl::exchange(other.begin_, nullptr))
        , end_(dpl::exchange(other.end_, nullptr)) {}

    constexpr dynamic_array& operator=(dynamic_array&& other) noexcept {
        destroy();
        begin_ = dpl::exchange(other.begin_, nullptr);
        end_ = dpl::exchange(other.end_, nullptr);
        return *this;
    }

    constexpr reference operator[](difference_type idx) noexcept {
        return begin_[idx];
    }

    constexpr const_reference operator[](difference_type idx) const noexcept {
        return begin_[idx];
    }

    constexpr pointer data() noexcept { return begin_; }
    constexpr const_pointer data() const noexcept { return begin_; }
    constexpr const_pointer cdata() const noexcept { return begin_; }
    constexpr pointer begin() noexcept { return begin_; }
    constexpr pointer end() noexcept { return end_; }
    constexpr const_pointer begin() const noexcept { return begin_; }
    constexpr const_pointer end() const noexcept { return end_; }
    constexpr const_pointer cbegin() const noexcept { return begin_; }
    constexpr const_pointer cend() const noexcept { return end_; }
    constexpr size_type size() const noexcept {
        return static_cast<size_type>(end_ - begin_);
    }
    constexpr bool empty() const noexcept { return size() == 0; }

    constexpr operator span<E>() noexcept { return span<E>(data(), size()); }

    constexpr operator span<E const>() const noexcept {
        return span<E const>(data(), size());
    }

private:
    constexpr void destroy() {
        auto* ptr = dpl::exchange(begin_, nullptr);
        end_ = nullptr;
        if (ptr != nullptr) {
            delete[] ptr;
        }
    }

    pointer begin_;
    pointer end_;
};

template <typename T, typename E>
constexpr dynamic_array<T> reinterpret_array(
    dynamic_array<E> const& src) noexcept
requires dpl::is_trivially_copyable_v<T> && dpl::is_trivially_copyable_v<E>
{
    static_assert(sizeof(T) > sizeof(E) ? sizeof(T) % sizeof(E) == 0
                                        : sizeof(E) % sizeof(T) == 0);
    dynamic_array<T> dst(src.size() * sizeof(E) / sizeof(T));
    if constexpr (sizeof(E) == sizeof(T)) {
        for (auto i = 0zu; i < dst.size(); ++i) {
            dst[i] = dpl::bit_cast<T>(src[i]);
        }
    } else if constexpr (sizeof(E) < sizeof(T)) {
        for (auto i = 0zu; i < dst.size(); ++i) {
            constexpr auto ratio = sizeof(T) / sizeof(E);
            dst[i] = [&]<size_t... Js>(dpl::index_sequence<Js...>) {
                return dpl::bit_cast<T>(test::array{src[i * ratio + Js]...});
            }(dpl::make_index_sequence<ratio>{});
        }
    } else {
        for (auto i = 0zu; i < src.size(); ++i) {
            constexpr auto ratio = sizeof(E) / sizeof(T);
            auto const tsrc = dpl::bit_cast<test::array<T, ratio>>(src[i]);
            for (auto j = 0zu; j < ratio; ++j) {
                dst[i * ratio + j] = tsrc[j];
            }
        }
    }

    return dst;
}

template <typename E>
span(dynamic_array<E>&) -> span<E>;
template <typename E>
span(dynamic_array<E> const&) -> span<E const>;
} // namespace support
} // namespace dpl::test
