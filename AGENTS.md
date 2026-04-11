# AI Rules for Flutter Package Development

This document defines the working rules for AI agents contributing to this Flutter OCR package.

The package follows **Domain-Driven Design (DDD)** to keep OCR concerns isolated, testable, and platform-agnostic.

---

# Core Assumptions

* The user is a senior engineer
* Favor direct, technical answers
* Do not explain basic Dart/Flutter concepts unless explicitly asked
* Optimize for correctness, clarity, and long-term maintainability

---

# Package Scope (Strict)

This is a **Flutter package/plugin**, not an application.

* Do NOT introduce:

  * Routing/navigation
  * Global app state
  * UI flows or screen-level abstractions
* Do NOT assume any app architecture (Bloc, Riverpod, etc.)
* Do NOT introduce service locators

Allowed:

* Small reusable widgets (if necessary)
* Pure Dart domain logic
* Platform abstractions

---

# Architectural Principles

## Domain-Driven Design (DDD)

Use DDD **only where it adds clarity**, not ceremony.

* **Domain Layer**

  * OCR entities, value objects, parsing logic
  * Immutable by default
  * No Flutter or platform dependencies

* **Application Layer (optional)**

  * Use-cases / orchestration logic
  * Thin and explicit

* **Infrastructure Layer**

  * Platform channels
  * File system, image processing, native bridges

* Keep all implementation details under `lib/src`

---

# Public API Design

* The ONLY public entrypoint is:

  * `lib/mobile_ocr.dart`

* Public API must be:

  * Minimal
  * Stable
  * Explicit

Rules:

* Do NOT leak internal types (`src/`)
* Prefer **value objects over raw maps**
* Prefer **immutable return types**
* Avoid exposing platform-specific details
* Every public API must have:

  * DartDoc comments
  * Clear input/output contracts
  * Defined error behavior

---

# API Evolution & Versioning

* Treat all public APIs as **semver-controlled**

* Breaking changes REQUIRE:

  * Clear justification
  * Migration guidance (in comments or docs)

* Additive changes are preferred over breaking changes

* Never:

  * Rename public APIs casually
  * Change return types without strong reason

---

# Platform & Plugin Rules

* Use `plugin_platform_interface` correctly
* Define a clear platform contract before implementation

Platform boundaries:

* Platform code must NOT leak into domain
* Platform responses must be mapped into domain models
* Handle platform inconsistencies explicitly

Error handling:

* Convert platform errors into **typed Dart exceptions**
* Do NOT pass raw platform exceptions to consumers

---

# Dependency Rules

* Prefer constructor injection
* No global state or hidden singletons
* Avoid service locators

Dependencies must:

* Solve a real package problem
* Be lightweight and maintained
* Not impose app-level constraints

---

# Concurrency & Performance

OCR is compute-heavy. Respect that.

* Use **isolates** for heavy processing when appropriate
* Avoid blocking the UI thread
* Minimize memory allocations in hot paths
* Avoid unnecessary object copying

---

# Dart Standards

* Use sound null safety

* Avoid `!` unless provably safe

* Prefer `async/await`

* Use:

  * Records
  * Pattern matching
  * Exhaustive `switch`

* Prefer explicit types at API boundaries

---

# Error Handling

* Never fail silently
* Throw **specific, typed exceptions**
* Preserve context (input, platform state, etc.)

Examples:

* Invalid input → `ArgumentError`
* Unsupported platform → custom exception
* OCR failure → domain-specific exception

---

# Logging

* Use `logging` package
* NEVER use `print`
* Logging must:

  * Be optional
  * Not pollute public API unless intentional

---

# Naming Conventions

| Type      | Convention   |
| --------- | ------------ |
| Classes   | `PascalCase` |
| Variables | `camelCase`  |
| Functions | `camelCase`  |
| Files     | `snake_case` |

* Avoid abbreviations unless domain-standard (e.g., OCR)

---

# Code Style

* Prefer composition over inheritance
* Keep functions small and focused
* Use `const` whenever possible
* Keep widgets immutable
* Avoid deep nesting

Comments:

* Required for public APIs
* Avoid redundant comments
* Explain **why**, not **what**

---

# File Structure

* Public API:

  * `lib/mobile_ocr.dart`

* Internal:

  * `lib/src/domain/...`
  * `lib/src/application/...`
  * `lib/src/infrastructure/...`
  * `lib/src/platform/...`

Keep structure **flat and discoverable**, not over-engineered.

---

# Testing Strategy

You MUST test at multiple layers:

### Domain

* Pure unit tests
* No Flutter dependency

### Infrastructure / Adapters

* Mock platform channels
* Validate mapping correctness

### Widgets (if any)

* Use `flutter_test`
* Keep minimal and focused

### Regression Tests

Required for:

* OCR parsing changes
* Model transformations
* Platform response mapping

---

# Backward Compatibility

* Preserve compatibility for published APIs
* If breaking:

  * Minimize blast radius
  * Document migration path

---

# Anti-Patterns (DO NOT DO)

* Introducing app architecture (Bloc, MVVM, etc.)
* Using service locators
* Leaking platform types into domain
* Returning `Map<String, dynamic>` instead of models
* Adding unnecessary abstraction layers
* Overusing inheritance
* Silent failures

---

# Key Rules Summary

1. Keep the package boundary small and strict
2. Model OCR as a clear domain
3. Keep platform concerns isolated
4. Prefer explicit dependencies
5. Treat `mobile_ocr.dart` as a stable contract
6. Optimize for testability and performance
7. Avoid app-level assumptions entirely
