# Copilot Instructions

## Code Style Guidelines

- **No conditional imports**: Always import modules at the top of the file. Never use conditional imports inside functions or conditional blocks.

- **No default field and function arguments**: Do not use default values for function parameters or dataclass fields. All arguments must be explicitly provided by the caller.
