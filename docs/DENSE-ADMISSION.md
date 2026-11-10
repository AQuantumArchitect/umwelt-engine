# Dense construction admission

Default-backend nodes fall back to cumulant before dense allocation when either
named limit is exceeded. `QuantumField.backend_fallbacks` and a warning record
the node, role count and reason. An explicitly pinned `qubit` node refuses instead
of changing the model silently. Direct `QubitCluster` construction uses the same
guard. Cumulant is an approximation, not a claim of dense-equivalent accuracy.

The role ceiling is `UMWELT_DENSE_MAX_QUBITS` (initial prior12). The additional
construction estimate budget is `UMWELT_DENSE_MAX_BYTES` (initial prior64MiB).
Provisional uncertainty widths are4roles and32MiB, respectively; neither is a
measured optimum or a host memory probe. These process-start settings are explicit
operator choices. No automatic learner currently changes them.

The byte estimate uses the actual evolve dtype and the constructor's matrix
inventory: `(6 + 7*n) * dtype_bytes * 4**n`. At the default complex64 precision,
12roles estimate11.25GiB, whereas8roles estimate31MiB. Merely limiting a single
matrix to12roles would therefore still allow an allocation far beyond a2GBhost.
The guard does not attempt the large allocation to discover that fact.

One local constructor measurement (eight dissipative roles, complex64, Python3.13
on the laptop) recorded28,476,434bytes peak with Python's tracemalloc, versus the
32,505,856byte estimate. That is a bounded check of this case, not a measurement
of the VM or a bound on every allocator/evolution path.

Objective: avoid allocation failures while retaining useful dense reference work.
Revision rule: record peak memory, total admitted clusters and subsequent task
outcomes on the actual host; compare a proposed setting with the retained prior
before adoption. Keep host admission and its reserve: BLAS workspace, cached
features, RK4 evolution, other clusters and other processes are not fully bounded
by this construction estimate. Increasing an environment limit does not establish
that the machine has the memory to run the work.
