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
- later stages (`promote_abi`, `target_abi`, `concat`) would need to repair invalid or undersized ABI choices
- concat target selection would become ambiguous or require repeated correction passes

Therefore, `common_abi` a new invariant must be introduced.

For fixed-width ABIs, the resulting ABI of `common_abi` must have size greater than or equal to all input ABIs.

This ensures:

- no information loss when normalizing heterogeneous SIMD operands
- the selected ABI can represent all inputs in a single domain
- `common_abi` behaves as a valid *join* under ABI capacity ordering (when such ordering exists)
- downstream promotion and concat stages operate on a stable ABI baseline

Since scalable ABIs have no compile-time width, if any input ABI is scalable, or the result ABI is scalable, this invariant is relaxed.

Formally, the invariant is given by the constraint:

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

### Stage 3 — Target ABI Selection (`target_abi`)

`target_abi` is an exposition-only metafunction used internally by the CPO. Given the set of operand ABIs, it computes the ABI that should hold the concatenated result. The existence of `target_abi` is a constraint of invoking the `concat` CPO.

`target_abi` is not a concrete type exposed to users. Its role is purely to route the call to the correct backend implementation. The selection is required to be deterministic and stable across translation units.

`target_abi<Ts...>` is an **exact-width ABI selection mechanism** derived through iterative promotion as follows:

1. `common_abi` is applied to unify operand ABIs.
2. `promote_abi` is applied to widen toward a candidate target.
3. Compare the total lane width of all operands and the lane width of `promote_abi_t`
   a. If the values are equal:
      - return the result of `promote_abi_t`

   b. If the lane width of `promote_abi_t` exceeds the total operand lane width:
      - error (no valid result exists)

   c. Otherwise:
      - update the candidate target to be `promote_abi_t` and repeat steps 2–3


### Stage 4 — Backend Dispatch (ADL)

Once the target ABI is resolved, the CPO invokes:

```c++
concat(target_abi<Ts...>{}, args...)
```

via ADL. This follows the standard DPL backend dispatch pattern: passing the ABI tag as the leading argument ensures lookup resolves into the namespace associated with the target ABI. Similarly with other opeartions, precedence is given to overloads involving custom simd-types. If no such overloads exist, all types are converted to `basic_simd` specializations and that overload is invoked.

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

`concat` is restricted to fixed-width ABIs for two reasons. First, the capacity constraint is verified statically, and the entire resolution pipeline depends on ABI widths being known at compile time. Second, scalable ABIs have only a single runtime-determined size — there is no wider scalable register to target, so the operation is not meaningful for them regardless of compile-time knowability.

### ABI Promotion Design

Two approaches to `promote_abi` were evaluated. The chosen design is described first.

#### Chosen: Exclusive Promotion

Promotion is defined by the **destination** ABI library, not the source. Each ABI that can receive concatenated values defines which smaller ABIs map into it — promotion is an incoming-edge relationship:

```
xmm → ymm   (defined by the ymm library)
ymm → zmm   (defined by the zmm library)
```

There is no global promotion chain owned by a single authority. The source ABI has no say in what it promotes to; the target ABI declares what it accepts. Multiple targets may independently define valid promotions from the same source without conflict, because each specializes `promote_abi` for its own incoming edges.

This means `promote_abi<xmm>` has exactly one answer within a given ABI family — whichever target has declared that incoming edge. A third-party ABI that wants a different widening path from `xmm` cannot re-specialize `promote_abi<xmm>` without causing a conflict. The escape hatch in that case is the wrapper ABI pattern (see [Extension](#extension)).

**Properties:**
- No promotion policy axis; no `ABI × context` space.
- Deterministic: for a given set of input ABIs, `target_abi` always resolves to the same result.
- Clean ownership: each ABI library owns how it accepts smaller ABIs; no library modifies another's definitions.
- Natural extensibility: new ABI families declare their own incoming edges without touching existing libraries.

**Accepted tradeoff:** A single ABI identity cannot simultaneously participate in two different promotion graphs. A third-party backend that needs a structurally different widening path from a canonical ABI (e.g. `xmm → vendor256` instead of `xmm → ymm`) must introduce a distinct ABI identity via wrapping or a fully custom type, rather than reusing the canonical one with a different policy.

---

#### Rejected: Promotion Policy

The alternative parameterized promotion by an explicit policy type:

```c++
promote_abi<A, Policy>
```

A default DPL policy would encode canonical promotions; third parties would define their own. `concat` would use the default policy, with `concat_with<Policy>` as an opt-in variant for explicit alternative selection.

**Advantages considered:**
- Supports multiple structural promotion interpretations per ABI identity without wrapper types.
- Separates ABI identity from promotion semantics cleanly in principle.

**Why it was rejected:**

The policy approach introduces a new concept solely to make promotion work within concat. It adds noticeable complexity for very little practical benefit. The expected number of third-party ABI extensions is small, and adding thrid-party ABI extensions is not the intended or recommended extension path for the system (see [Custom SIMD Types](#custom-simd-types-preferred-extension-model)).

---

## Extension

### ABI Extension

The number of ABI families DPL needs to reason about is expected to be small, so no ABI policy or context dispatch layer is introduced. Third parties extending the ABI graph should provide appropriate specializations of `common_abi` and `promote_abi` for their types. The wrapper ABI pattern is one supported mechanism:

```c++
// Wrapper ABI: introduces a distinct ABI identity
struct myxmm : xmm {};
```

A wrapper ABI is treated as a separate ABI family and does not modify the canonical promotion behavior of the base ABI. This pattern requires that the base ABI type is not marked `final`; a `final` base class cannot be derived from, so the wrapper trick is unavailable in that case.

Because ADL includes the namespaces of all direct and indirect base classes of an argument type, operations defined for the base ABI remain reachable through the wrapper. A call such as `add(myxmm{}, a, b)` will find `add(xmm, ...)` overloads via ADL if no `myxmm`-specific overload exists, and `myxmm` converts implicitly to `xmm` via derived-to-base. The wrapper therefore inherits the full operation set of the base ABI without any additional definitions, while still introducing an independent promotion identity.

### Custom SIMD Types (Preferred Extension Model)

Third-party systems are encouraged to define custom SIMD types rather than extending the ABI graph directly. Custom types integrate via ADL-based dispatch, do not require modification of the `common_abi`/`promote_abi` graph, and do not compete with canonical promotion rules.

DPL treats non-primary SIMD types (i.e. types that are not specializations of `basic_simd`) as first-class extension points. See the [extensibility documentation](./extensibility.md) for the full dispatch and fallback model.

---

## Summary

```
concat(x0, x1, ...)
    │
    ▼
common_abi          ← unify operand ABI domains
    │
    ▼
promote_abi         ← find widening path (optional, may repeat)
    │
    ▼
target_abi          ← select destination ABI
    │
    ▼
concat(target_abi<Ts...>{}, x0, x1, ...)   ← ADL backend dispatch
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
