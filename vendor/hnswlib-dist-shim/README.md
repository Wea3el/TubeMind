# hnswlib distribution shim

Fast GraphRAG depends on the `hnswlib` distribution, but the upstream package
does not currently publish a Windows CPython 3.12 wheel and requires local C++
build tools. `chroma-hnswlib==0.7.6a4` publishes a compatible wheel that exposes
the same `hnswlib` import module.

This empty package satisfies the dependency resolver while leaving the actual
`hnswlib` module to be provided by `chroma-hnswlib`.
