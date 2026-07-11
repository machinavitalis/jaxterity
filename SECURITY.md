# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately via GitHub's **Report a vulnerability** flow:
**Security → Advisories → Report a vulnerability**
(`https://github.com/machinavitalis/jaxterity/security/advisories/new`).

Include the affected version/commit, a minimal reproduction, and the impact you
observed. We aim to acknowledge within a few business days.

## Supported versions

Security fixes target the latest `main` and the most recent tagged release.
Older tags are not maintained.

## Scope notes for users

Some behaviors are insecure **by design** if misused — not vulnerabilities, but
know what you're running:

- **Untrusted robot descriptions.** Importing a URDF/MJCF (and any FMU or
  external model it references) parses and can execute third-party assets. Treat
  a robot description from an unknown source like untrusted input.
- **Telemetry / calibration inputs.** Calibration consumes external data files;
  only fit against telemetry you trust.

If you can escalate beyond these documented behaviors (e.g. code execution from
a path documented as safe), that *is* a vulnerability — please report it.
