// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/addsub.h"
#include "dpl/core/operations/arithmetic/fmadd.h"
#include "dpl/core/operations/arithmetic/fmaddsub.h"
#include "dpl/core/operations/arithmetic/fmsub.h"
#include "dpl/core/operations/arithmetic/fmsubadd.h"
#include "dpl/core/operations/arithmetic/fnmadd.h"
#include "dpl/core/operations/arithmetic/fnmsub.h"
#include "dpl/core/operations/arithmetic/multiply.h"
#include "dpl/core/operations/arithmetic/subadd.h"
#include "dpl/core/operations/arithmetic/subtract.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

/**
 * Computes a*b + c, fused if available, unfused (a*b + c) otherwise.
 *
 * Convenience CPO: no canonical implementation, no unqualified-ADL
 * extension point. The only injectable hook is extended<muladd_t{}>.
 * No rounding guarantee -- use fmadd directly if single-rounding is required.
 *
 * @see fmadd for the strict, guaranteed-fused equivalent.
 */
struct DPL_EMPTY_BASES muladd_t :
    public arithmetic_base<muladd_t>,
    public maskable_accumulation_base<muladd_t>,
    public ternary_broadcastable_operation<muladd_t> {
    using operation_base<muladd_t>::operator();
    using maskable_accumulation_base<muladd_t>::operator();
    using ternary_broadcastable_operation<muladd_t>::operator();
};

struct DPL_EMPTY_BASES mulacc_t :
    public arithmetic_base<mulacc_t>,
    public maskable_accumulation_base<mulacc_t>,
    public ternary_broadcastable_operation<mulacc_t> {
    using operation_base<mulacc_t>::operator();
    using maskable_accumulation_base<mulacc_t>::operator();
    using ternary_broadcastable_operation<mulacc_t>::operator();
};

struct DPL_EMPTY_BASES mulsub_t :
    public arithmetic_base<mulsub_t>,
    public maskable_accumulation_base<mulsub_t>,
    public ternary_broadcastable_operation<mulsub_t> {
    using operation_base<mulsub_t>::operator();
    using maskable_accumulation_base<mulsub_t>::operator();
    using ternary_broadcastable_operation<mulsub_t>::operator();
};

struct DPL_EMPTY_BASES mulsac_t :
    public arithmetic_base<mulsac_t>,
    public maskable_accumulation_base<mulsac_t>,
    public ternary_broadcastable_operation<mulsac_t> {
    using operation_base<mulsac_t>::operator();
    using maskable_accumulation_base<mulsac_t>::operator();
    using ternary_broadcastable_operation<mulsac_t>::operator();
};

struct DPL_EMPTY_BASES nmuladd_t :
    public arithmetic_base<nmuladd_t>,
    public maskable_accumulation_base<nmuladd_t>,
    public ternary_broadcastable_operation<nmuladd_t> {
    using operation_base<nmuladd_t>::operator();
    using maskable_accumulation_base<nmuladd_t>::operator();
    using ternary_broadcastable_operation<nmuladd_t>::operator();
};

struct DPL_EMPTY_BASES nmulacc_t :
    public arithmetic_base<nmulacc_t>,
    public maskable_accumulation_base<nmulacc_t>,
    public ternary_broadcastable_operation<nmulacc_t> {
    using operation_base<nmulacc_t>::operator();
    using maskable_accumulation_base<nmulacc_t>::operator();
    using ternary_broadcastable_operation<nmulacc_t>::operator();
};

struct DPL_EMPTY_BASES nmulsub_t :
    public arithmetic_base<nmulsub_t>,
    public maskable_accumulation_base<nmulsub_t>,
    public ternary_broadcastable_operation<nmulsub_t> {
    using operation_base<nmulsub_t>::operator();
    using maskable_accumulation_base<nmulsub_t>::operator();
    using ternary_broadcastable_operation<nmulsub_t>::operator();
};

struct DPL_EMPTY_BASES nmulsac_t :
    public arithmetic_base<nmulsac_t>,
    public maskable_accumulation_base<nmulsac_t>,
    public ternary_broadcastable_operation<nmulsac_t> {
    using operation_base<nmulsac_t>::operator();
    using maskable_accumulation_base<nmulsac_t>::operator();
    using ternary_broadcastable_operation<nmulsac_t>::operator();
};

struct DPL_EMPTY_BASES muladdsub_t :
    public arithmetic_base<muladdsub_t>,
    public maskable_accumulation_base<muladdsub_t>,
    public ternary_broadcastable_operation<muladdsub_t> {
    using operation_base<muladdsub_t>::operator();
    using maskable_accumulation_base<muladdsub_t>::operator();
    using ternary_broadcastable_operation<muladdsub_t>::operator();
};

struct DPL_EMPTY_BASES muladdsac_t :
    public arithmetic_base<muladdsac_t>,
    public maskable_accumulation_base<muladdsac_t>,
    public ternary_broadcastable_operation<muladdsac_t> {
    using operation_base<muladdsac_t>::operator();
    using maskable_accumulation_base<muladdsac_t>::operator();
    using ternary_broadcastable_operation<muladdsac_t>::operator();
};

struct DPL_EMPTY_BASES mulsubadd_t :
    public arithmetic_base<mulsubadd_t>,
    public maskable_accumulation_base<mulsubadd_t>,
    public ternary_broadcastable_operation<mulsubadd_t> {
    using operation_base<mulsubadd_t>::operator();
    using maskable_accumulation_base<mulsubadd_t>::operator();
    using ternary_broadcastable_operation<mulsubadd_t>::operator();
};

struct DPL_EMPTY_BASES mulsubacc_t :
    public arithmetic_base<mulsubacc_t>,
    public maskable_accumulation_base<mulsubacc_t>,
    public ternary_broadcastable_operation<mulsubacc_t> {
    using operation_base<mulsubacc_t>::operator();
    using maskable_accumulation_base<mulsubacc_t>::operator();
    using ternary_broadcastable_operation<mulsubacc_t>::operator();
};

template <>
struct operation_signature<muladd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<mulacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<mulsub_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<mulsac_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<nmuladd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<nmulacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<nmulsub_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<nmulsac_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<muladdsub_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<muladdsac_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<mulsubadd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};
template <>
struct operation_signature<mulsubacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<muladd_t> : ternary_broadcasting_fallback<muladd_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(a, b, c);
        } else {
            return dx::add(dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(a, mask, b, c);
        } else {
            return dx::add(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(a, mask, b, c);
        } else {
            return dx::add(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(a, mask, b, c);
        } else {
            return dx::add(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmadd)) {
            return dx::fmadd(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, dx::multiply(a, b), c);
        }
    }

    using ternary_broadcasting_fallback<muladd_t>::operator();
};

template <>
struct fallback_impl<mulacc_t> : ternary_broadcasting_fallback<mulacc_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(a, b, c);
        } else {
            return dx::add(a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(a, mask, b, c);
        } else {
            return dx::add(a, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(a, mask, b, c);
        } else {
            return dx::add(a, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(a, mask, b, c);
        } else {
            return dx::add(a, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_mask<M> || extended_vector<AT> ||
        extended_vector<BT> || extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, a, dx::multiply(b, c));
        }
    }

    using ternary_broadcasting_fallback<mulacc_t>::operator();
};

template <>
struct fallback_impl<mulsub_t> : ternary_broadcasting_fallback<mulsub_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, b, c);
        } else {
            return dx::subtract(dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(a, b), c);
        }
    }

    using ternary_broadcasting_fallback<mulsub_t>::operator();
};

template <>
struct fallback_impl<mulsac_t> : ternary_broadcasting_fallback<mulsac_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(a, b, c);
        } else {
            return dx::subtract(dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsac)) {
            return dx::fmsac(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(b, c), a);
        }
    }

    using ternary_broadcasting_fallback<mulsac_t>::operator();
};

template <>
struct fallback_impl<nmuladd_t> : ternary_broadcasting_fallback<nmuladd_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(a, b, c);
        } else {
            return dx::subtract(c, dx::multiply(a, b));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, c, dx::multiply(a, b));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, c, dx::multiply(a, b));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, c, dx::multiply(a, b));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, c, dx::multiply(a, b));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, c, dx::multiply(a, b));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmadd)) {
            return dx::fnmadd(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, c, dx::multiply(a, b));
        }
    }

    using ternary_broadcasting_fallback<nmuladd_t>::operator();
};

template <>
struct fallback_impl<nmulacc_t> : ternary_broadcasting_fallback<nmulacc_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(a, b, c);
        } else {
            return dx::subtract(a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, a, dx::multiply(b, c));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmacc)) {
            return dx::fnmacc(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, a, dx::multiply(b, c));
        }
    }

    using ternary_broadcasting_fallback<nmulacc_t>::operator();
};

template <>
struct fallback_impl<nmulsub_t> : ternary_broadcasting_fallback<nmulsub_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(a, b, c);
        } else {
            return dx::negate(dx::add(c, dx::multiply(a, b)));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(a, mask, b, c);
        } else {
            return dx::negate(a, mask, dx::add(c, dx::multiply(a, b)));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(a, mask, b, c);
        } else {
            return dx::negate(a, mask, dx::add(c, dx::multiply(a, b)));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(a, mask, b, c);
        } else {
            return dx::negate(a, mask, dx::add(c, dx::multiply(a, b)));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(zero, mask, a, b, c);
        } else {
            return dx::negate(a, mask, dx::add(c, dx::multiply(a, b)));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(zero, mask, a, b, c);
        } else {
            return dx::negate(zero, mask, dx::add(c, dx::multiply(a, b)));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsub)) {
            return dx::fnmsub(zero, mask, a, b, c);
        } else {
            return dx::negate(zero, mask, dx::add(c, dx::multiply(a, b)));
        }
    }

    using ternary_broadcasting_fallback<nmulsub_t>::operator();
};

template <>
struct fallback_impl<nmulsac_t> : ternary_broadcasting_fallback<nmulsac_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(a, b, c);
        } else {
            return dx::negate(dx::add(a, dx::multiply(b, c)));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(a, mask, b, c);
        } else {
            return dx::negate(a, mask, dx::add(a, dx::multiply(b, c)));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(a, mask, b, c);
        } else {
            return dx::negate(a, mask, dx::add(a, dx::multiply(b, c)));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(a, mask, b, c);
        } else {
            return dx::negate(a, mask, dx::add(a, dx::multiply(b, c)));
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(zero, mask, a, b, c);
        } else {
            return dx::negate(zero, mask, dx::add(a, dx::multiply(b, c)));
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(zero, mask, a, b, c);
        } else {
            return dx::negate(zero, mask, dx::add(a, dx::multiply(b, c)));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fnmsac)) {
            return dx::fnmsac(zero, mask, a, b, c);
        } else {
            return dx::negate(zero, mask, dx::add(a, dx::multiply(b, c)));
        }
    }

    using ternary_broadcasting_fallback<nmulsac_t>::operator();
};

template <>
struct fallback_impl<muladdsub_t> : ternary_broadcasting_fallback<muladdsub_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, b, c);
        } else {
            return dx::addsub(dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsub)) {
            return dx::fmaddsub(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(a, b), c);
        }
    }

    using ternary_broadcasting_fallback<muladdsub_t>::operator();
};

template <>
struct fallback_impl<muladdsac_t> : ternary_broadcasting_fallback<muladdsac_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, b, c);
        } else {
            return dx::addsub(dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, mask, b, c);
        } else {
            return dx::addsub(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_mask<M> || extended_vector<AT> ||
        extended_vector<BT> || extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(zero, mask, a, b, c);
        } else {
            return dx::addsub(zero, mask, dx::multiply(b, c), a);
        }
    }

    using ternary_broadcasting_fallback<muladdsac_t>::operator();
};

template <>
struct fallback_impl<mulsubadd_t> : ternary_broadcasting_fallback<mulsubadd_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(a, b, c);
        } else {
            return dx::subadd(dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubadd)) {
            return dx::fmsubadd(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(a, b), c);
        }
    }

    using ternary_broadcasting_fallback<mulsubadd_t>::operator();
};

template <>
struct fallback_impl<mulsubacc_t> : ternary_broadcasting_fallback<mulsubacc_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, b, c);
        } else {
            return dx::subadd(dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_mask<M> || extended_vector<AT> ||
        extended_vector<BT> || extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmaddsac)) {
            return dx::fmaddsac(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(b, c), a);
        }
    }

    using ternary_broadcasting_fallback<mulsubacc_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::muladd_t muladd{};
inline constexpr internal::mulacc_t mulacc{};
inline constexpr internal::mulsub_t mulsub{};
inline constexpr internal::mulsac_t mulsac{};
inline constexpr internal::nmuladd_t nmuladd{};
inline constexpr internal::nmulacc_t nmulacc{};
inline constexpr internal::nmulsub_t nmulsub{};
inline constexpr internal::nmulsac_t nmulsac{};
inline constexpr internal::muladdsub_t muladdsub{};
inline constexpr internal::muladdsac_t muladdsac{};
inline constexpr internal::mulsubadd_t mulsubadd{};
inline constexpr internal::mulsubacc_t mulsubacc{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
