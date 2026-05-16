# Design Document: `concat`

## Overview

`concat` is a variadic SIMD concatenation operation that combines two or more fixed-width SIMD values into a single wider SIMD value. The resulting type has an ABI whose register width equals the sum of the inputs' widths.

```c++
// Two xmm (128-bit) → ymm (256-bit)
basic_simd<float, xmm> a, b;
auto c = concat(a, b); // basic_simd<float, ymm>

// Four xmm (128-bit) → zmm (512-bit)
basic_simd<float, xmm> a, b, c, d;
auto z = concat(a, b, c, d); // basic_simd<float, zmm>

// Mixed ABI: xmm + ymm + xmm (128 + 256 + 128 = 512-bit) → zmm
basic_simd<float, xmm> lo, hi;
basic_simd<float, ymm> mid;
auto z = concat(lo, mid, hi); // basic_simd<float, zmm>
```

`concat` is only defined for fixed-width ABI types. Scalable ABIs are excluded for 2 reasons: their lane count is not known until runtime, making static capacity reasoning impossible; and more fundamentally, a scalable ABI has only a single size — there is no notion of a "wider" scalable register to concatenate into, so the operation has no meaningful definition for them.

---

## Public API

`concat` is exposed as a variadic Customization Point Object (CPO):

```c++
concat(x0, x1, ...);
```

The call site is fully type-erased with respect to ABI. No ABI types appear in user code, and all resolution logic is hidden inside the CPO. This is consistent with the rest of DPL's public interface.

---

## Resolution Pipeline

A call to `concat(x0, x1, ...)` is resolved through four sequential stages.

### Stage 1 — ABI Compatibility Resolution (`common_abi`)

Before any widening can occur, the operands must share a compatible ABI domain. `common_abi` unifies operand ABIs, analogous to `std::common_type` for arithmetic types:

```c++
common_abi<xmm, xmm> → xmm
common_abi<xmm, ymm> → ymm
```

This step is a no-op when all operands already share the same ABI, and is required to handle mixed-ABI inputs in a well-defined way.

`common_abi` is defined by ABI libraries, following the same extension model as the rest of DPL: user-defined ABI families may specialize it for their own types.

While `common_abi` is conceptually a *compatibility trait*, in practice it is also used as the **first normalization step before promotion and concatenation**.

This introduces an important requirement:

> The resulting ABI must never be "smaller" than any of its inputs in representational capacity.

Without this guarantee:
- mixed-ABI expressions could silently downgrade capacity
- later stages of the concatenation would need to repair invalid or undersized ABI choices
- concat target selection would become ambiguous or require repeated correction passes

Therefore, `common_abi` a new invariant must be introduced.

For fixed-width ABIs, the resulting ABI of `common_abi` must have size greater than or equal to all input ABIs.

This ensures:

- no information loss when normalizing heterogeneous SIMD operands
- the selected ABI can represent all inputs in a single domain
- `common_abi` behaves as a valid *join* under ABI capacity ordering (when such ordering exists)
- downstream promotion and concat stages operate on a stable ABI baseline

Since scalable ABIs have no compile-time width, if any input ABI is scalable, or the result ABI is scalable, this invariant is relaxed.

Formally, the invariant is given by the following constraint:

```c++
(scalable_abi<common_abi_t<Ts...>> || ... || scalable_abi<Ts>) || common_abi_t<Ts...>::size >= max(Ts::size...)
```

### Stage 2 — ABI Promotion (`promote_abi`)

`promote_abi` models the canonical single-step widening within a fixed-width ABI family:

```c++
promote_abi<xmm> → ymm
promote_abi<ymm> → zmm
```

In the context of the concat pipeline, `promote_abi` provides the structural expansion path when operands must be packed into a wider register. It answers the question: *given the unified input ABI, what is the next wider ABI?*

### Stage 3 — Target ABI Selection

For target selection, an exposition-only metafunction, `concat-target-abi`, used internally by the CPO. Given the set of operand ABIs, it computes the ABI that should hold the concatenated result. The existence of `concat-target-abi` is a constraint of invoking the `concat` CPO. Its role is purely to route the call to the correct backend implementation. The selection is required to be deterministic and stable across translation units (TU).

`concat-target-abi<Ts...>` is derived through iterative promotion as follows:

1. `common_abi` is applied to unify operand ABIs.
2. `promote_abi` is applied to widen toward a candidate target.
3. Compare the total lane width of all operands and the lane width of `promote_abi`
   a. If the values are equal:
      - return the result of `promote_abi`

   b. If the lane width of `promote_abi` exceeds the total operand lane width:
      - error (no valid result exists)

   c. Otherwise:
      - update the candidate target to be `promote_abi_t` and repeat steps 2–3


### Stage 4 — Backend Dispatch (ADL)

Once the target ABI is resolved, the CPO invokes either:

```c++
concat<common_abi_t<Ts...>>(concat_target_abi<Ts...>{}, args...)
```

or

```c++
concat<concat_target_abi<Ts...>>(common_abi_t<Ts...>{}, args...)
```

via ADL, with the latter taking precedence (see [docs](../include/dpl/core/operations/concat.h)).

The backend implementation is then responsible for the actual register packing, lane layout, and any architecture-specific instruction selection. As with most fixed_width operations, a fallback is provided. However, unlike other operations, `concat`'s fallback is `consteval`, which means the fallback is only used if the a runtime concatenation is defined.

## Responsibility Split

| Layer | Components | Responsibilities |
|---|---|---|
| Public API | CPO (`concat`) | User-facing simplicity, type erasure, dispatch into the resolution pipeline |
| ABI Resolution | `common_abi`, `promote_abi`, `target_abi` | Compatibility unification, widening path, target selection, capacity reasoning |
| Backend | `concat(target_abi<Ts...>, args...)` | SIMD construction, lane packing, architecture-specific instruction selection |


---

## Design Decisions

### Fixed-Width Only

`concat` is restricted to fixed-width ABIs for two reasons. First, the capacity constraint is verified statically, and the entire resolution pipeline depends on ABI widths being known at compile time. Second, scalable ABIs have only a single runtime-determined size, there is no wider scalable register to target, so the operation is not meaningful for them regardless of compile-time knowledge.

### ABI Promotion Design

Two approaches to `promote_abi` were evaluated. The chosen design is described first.

#### Exclusive Promotion (Chosen)

In this approach, each `promote_abi` has exactly one answer within a given ABI family. 

Promotion can be defined by the destination or the source ABI library. In practice, it is usually the destination library since Instruction Set Architecture (ISA) with larger registers are typically extensions of ISAs with smaller register. This is so that modularity and extensibility of ABI libraries is maintained, where extensions can define the promotion without modifying the extended library.

A simple example are a set of modular libraries the implement the backend for Intel's x86 SIMD intrinsics:

```
xmm
xmm → ymm   (defined by the ymm library)
ymm → zmm   (defined by the zmm library)
```

Starting with a standalone `xmm` ABI, a separate `ymm` library is defined and a promotion path is defined from `xmm` to `ymm` in the `ymm` library. The same can be done for the `zmm` library.

However, a single ABI identity cannot simultaneously participate in two different promotion graphs. A third-party backend that needs a structurally different widening path from a canonical ABI (e.g. `xmm → vendor256` instead of `xmm → ymm`) must introduce a distinct ABI identity via wrapping or a fully custom type (see [Extension](#extension)), rather than reusing the canonical one.

##### Implementation: Stateful Friend Injection

With the exclusive promotion model chosen, the remaining question is how `promote_abi` (and its inverse) is implemented. Two mechanisms are available: conventional template specialization and stateful friend function injection. The latter is chosen.

ABI promotion edges are registered by instantiating `define_promotion<Src, Target>()`:

```c++
namespace internal {

template <typename To, typename From> // Exposition only
concept valid_promotion_from = fixed_width_abi<To> && fixed_width_abi<From> &&
    requires { requires To::size > From::size; } && common_abi_with<To, From>;

template<typename T>
struct promotion {
    friend consteval auto promote(promotion<T>) noexcept;
};

template<typename T>
struct demotion {
    friend consteval auto demote(demotion<T>) noexcept;
};

template<fixed_width_abi T, valid_promotion_from<T> U>
struct define_promotion_t {
    friend consteval auto promote(promotion<T>) noexcept {
        return std::type_identity<U>{};
    }
    friend consteval auto demote(demotion<U>) noexcept{
        return std::type_identity<T>{};
    }
};

} // namespace internal

template<fixed_width_abi T, valid_promotion_from<T> U>
consteval auto define_promotion() noexcept {
    return sizeof(internal::define_promotion_t<T, U>) > 0;
}

template<typename T>
using promote_abi_t = typename decltype(promote(internal::promotion<T>{}))::type;

template<typename T>
using demote_abi_t  = typename decltype(demote(internal::demotion<T>{}))::type;
```

> [!NOTE]
> The above is just an example implementation, DPL's actual implementation may differ slightly

ABI libraries register edges at namespace scope:

```c++
// In the ymm library header:
static_assert(define_promotion<xmm, ymm>());

// In the zmm library header:
static_assert(define_promotion<ymm, zmm>());
```

Each `define_promotion<Src, Target>()` instantiation injects two friend functions into the `internal` namespace: `promote(promotion<Src>)` returning `Target`, and `demote(demotion<Target>)` returning `Src`. Both become discoverable via ADL from that point forward in the Translation Unit (TU). For C++23 (minimum requirement of DPL), the `static_assert` can be hidden behind a macro, e.g. `DPL_DEFINE_ABI_PROMOTION` for brevity, but in C++26 `consteval` block can be used instead:

```c++
consteval {
   define_promotion<myabi<128>, myabi<256>>();
}
```

##### Why not template specialization?

Template specialization is the more familiar mechanism:

```c++
template<> struct promote_abi<xmm> { using type = ymm; };
template<> struct promote_abi<ymm> { using type = zmm; };
```

It works correctly for simple cases, but it has a structural limitation: it is unidirectional. Defining a promotion from `xmm` to `ymm` does not automatically imply a demotion from `ymm` to `xmm`. A `split` API (the inverse of `concat`) needs `demote_abi_t`, which would require ABI library authors to maintain both `promote_abi` and `demote_abi` specializations independently, with no enforcement that they remain consistent. The alternative with be inroducing an conceptually asymetric API for `split` and `concat`, whereby `split<ABI>` the destrination ABI needs to be specified through the `split` function as opposed to a simpler mechanism like `split<N>`, where `N` is the number of results to split into.

The friend injection mechanism eliminates this entirely. A single `define_promotion<xmm, ymm>()` call populates both directions atomically; the promotion and demotion graphs are always in sync by construction.

##### Uniqueness constraint

Because `define_promotion<Src, Target>()` injects `promote(promotion<Src>)` as a friend function, instantiating it for the same `Src` with two different targets produces a duplicate function definition — a hard compile error in any TU that includes both registrations. This enforces at the language level that each source ABI has at most one promotion target within a given TU, which is the uniqueness property the exclusive promotion model requires. 

A third-party ABI that wants a different widening path from a canonical source (e.g. `xmm → vendor256` instead of `xmm → ymm`) therefore cannot call `define_promotion<xmm, vendor256>()` without conflicting with the canonical registration. The escape hatch is the wrapper ABI pattern (see [Extension](#extension)), which introduces a new source identity rather than competing over an existing one.

##### Pattern matching for parametric ABI families

Template specialization allows natural pattern matching over parametric ABI families:

```c++
template<size_t N>
struct promote_abi<myabi<N>> { using type = myabi<(N << 1)>; };
```

Friend injection requires explicit enumeration instead, since partial specialization of a class template is not involved:

```c++
// Option 1: hardcode the known sizes (preferred for small graphs)
static_assert(define_promotion<myabi<128>, myabi<256>>());
static_assert(define_promotion<myabi<256>, myabi<512>>());

// Option 2: generate via index_sequence for numerical patterns
static_assert([]<size_t... Is>(std::index_sequence<Is...>) {
    return (define_promotion<myabi<(128 << Is)>, myabi<(128 << Is + 1)>>() && ...);
}(std::make_index_sequence<2>{}));

// C++26 using dpl::index_sequence
consteval {
template for (auto idx : dpl::make_index_sequence<2>{}) {
    define_promotion<myabi<(128 << idx)>, myabi<(128 << idx + 1)>>();
}
}
```

This is a real ergonomic regression for parametric families, but it is not a practical concern for SIMD ABI graphs. The x86 family has accumulated three register sizes over twenty years. Explicit enumeration is not only tractable — it is arguably more readable than a template pattern that requires readers to mentally evaluate the size expression.

For non-numerical parametric patterns, the enumeration approach still applies, though the `static_assert` method must be used where needed to force instantiation without a clean numerical sequence.

##### On the well-formedness of stateful TMP

Stateful metaprogramming via injected friend functions is a gray-area in terms of whether it is classified as undefined behaviour. The main concern is whether or not this would introduce ODR violations. In the context of this design, that concern does not apply beyond what already exists for template specialization.

Consider the two failure modes, which are symmetric across both mechanisms:

| Scenario | Template specialization | Friend injection |
|---|---|---|
| TUs include disjoint promotion headers/modules | Silent ODR violation — `promote_abi_t<T>` differs per TU | Same — ADL resolves differently per TU |
| TUs include conflicting promotion headers/modules | Hard error — duplicate specialization | Hard error — duplicate friend definition |
| TUs include consistent promotion headers/modules | Well-defined | Well-defined |

The risk profile is identical. Both mechanisms rely on the same discipline: the promotion graph must be consistently defined across all translation units. Neither provides stronger guarantees than the other in the face of misuse.

It is also worth noting that stateful metaprogramming is no longer an exotic technique. C++26 formalizes the concept via the reflection API (`std::meta::define_class`) and `consteval` blocks, which provides first-class mechanisms for mutating compile-time state. The friend injection pattern is a pre-C++26 approximation of the same idea, though very much less powerful.

---

#### Rejected: Multi-Promotion via Policy

The alternative parameterized promotion by an explicit policy type:

```c++
promote_abi<A, Policy>
```

A default DPL policy would encode canonical promotions; third parties would define their own. `concat` would use the default policy, with `concat_with<Policy>` as an opt-in variant for explicit alternative selection.

**Advantages considered:**
- Supports multiple structural promotion interpretations per ABI identity without wrapper types.
- Separates ABI identity from promotion semantics cleanly in principle.

**Why it was rejected:**

The policy approach introduces a new concept solely to make promotion work. It adds noticeable complexity for very little practical benefit. The expected number of third-party ABI extensions is small, and adding thrid-party ABI extensions is not the intended or recommended extension path for the system (see [Custom SIMD Types](#custom-simd-types-preferred-extension-model)). In addition, even more complexity must be added to produce a bidirectional promotion graph.

---

## Extension

### ABI Extension (Advanced / Low-Level)

DPL assumes that ABI promotion forms a deterministic directed graph, where each source ABI has a unique promotion path toward a given target ABI resolution. In other words, for a given ABI identity, the promotion behavior is not intended to be ambiguous or context-dependent.

Because of this constraint, changing the result of ABI promotion (for example, redirecting a set of SIMD values from one target ABI to another) requires introducing a new source ABI identity, rather than modifying existing promotion rules.

A supported mechanism for this is the wrapper ABI pattern:

```c++
// Wrapper ABI: introduces a distinct ABI identity
struct myxmm : xmm {};
```

A wrapper ABI creates a new ABI identity while preserving compatibility with the base ABI. It becomes a distinct node in the ABI promotion graph, allowing it to define an independent promotion path without modifying canonical ABI behavior.

Because the wrapper inherits from the base ABI, it also preserves ADL-based dispatch. Using the example above, operations defined for `xmm` remain usable for `myxmm` without reimplementation, unless explicitly overridden.

However, this pattern is **not recommended for general use**.

While it provides a mechanism for redirecting promotion behavior, it introduces several potential issues:

* **User confusion:** most users do not directly interact with ABI types, making wrapper ABIs unintuitive at the usage level.
* **Competing backend identity:** wrapper ABIs effectively introduce a parallel ABI hierarchy. In the best case, this remains invisible to ADL dispatch; however, in more complex cases it can increase the difficulty of reasoning about which promotion and dispatch paths are actually being taken, making ABI resolution behavior harder to trace and debug.
* **Error complexity:** ABI resolution failures may surface as confusing compile-time errors across the library.
* **Graph contamination risk:** users may accidentally mix wrapper and canonical ABIs, leading to inconsistent or incompatible ABI graph states.

For these reasons, wrapper ABIs are considered an advanced escape hatch rather than a primary extension strategy.

### Custom SIMD Types (Preferred Extension Model)

Custom SIMD types are the preferred extension mechanism in DPL.

Library consumers are expected to use a single coherent ABI implementation set per backend family within a codebase. This does not mean restricting usage to a single ABI width (e.g. Intel 128-bit, 256-bit, 512-bit). Multiple ABI modules that belong to the same designed family and are intended to interoperate may be used together.

However, introducing multiple competing ABI implementations for the same backend concept (e.g. two independent Intel 128-bit ABI implementations that either do not share a common promotion model or have a forced promotion model) is discouraged. Mixing such implementations can lead to ambiguity in promotion and resolution rules and makes ABI behavior harder to reason about.

In practice, most user-facing code should not need to interact with ABI selection directly. Instead, behavior is expressed through custom SIMD types and CPOs, while ABI selection and promotion are handled internally by the chosen library implementation (including rules such as `common_abi` and `promote_abi`).

Custom SIMD types therefore provide the primary extension path:

* they integrate via ADL-based customization point objects (CPOs)
* they do not participate in ABI transformations, e.g. `common_abi` and `promote_abi`
* they avoid interaction with ABI-level resolution rules
* they keep extension localized to user-defined types rather than global ABI structure

This design keeps ABI-level complexity internal to the library, while allowing users to extend functionality through types that participate naturally in existing DPL operations.

---

## Summary

```
concat(x0, x1, ...)
    │
    ▼
common_abi          ← unify operand ABI domains
    │
    ▼
concat-target-abi          ← select destination ABI
    │
    ▼
concat(concat-target-abi<Ts...>{}, x0, x1, ...)   ← ADL backend dispatch
    │
    ▼
SIMD construction   ← architecture-specific lane packing
```

### Key Properties

- Public API is CPO-based and exposes no ABI types to the call site.
- ABI resolution logic is entirely internal to the CPO.
- Capacity must match exactly; oversubscription is a compile-time error.
- Target ABI selection is deterministic and stable across translation units.
- Promotion is target-driven (exclusive); promotion policies are intentionally excluded.
- Extension is achieved via custom SIMD types or wrapper ABI identities.
- `concat` is only valid for fixed-width ABI types.
