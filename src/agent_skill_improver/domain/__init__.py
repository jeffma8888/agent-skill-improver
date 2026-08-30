"""Pure domain layer.

Nothing in this package performs I/O, spawns a process, or imports from
``adapters`` / ``app``. Dependencies point inward: adapters depend on domain,
never the reverse. That is what makes the safety invariants testable without a
model, a network, or a real skill tree.
"""
