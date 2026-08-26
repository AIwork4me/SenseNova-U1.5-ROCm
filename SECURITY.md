# Security Policy

## Supported versions
The tip of `main` only.

## Reporting a vulnerability
Please do NOT open a public issue. Use GitHub's private vulnerability
reporting ("Security" tab → "Report a vulnerability"). Include
reproduction steps and affected components (scripts/, patches/, docs/
pipeline). Expect a response within 7 days.

## Scope
This repo ships shell/python scripts and upstream patches; it does not
serve anything to the network. Model weights come from upstream
ModelScope — weight integrity is verified by
configs/artifact-manifest.json SHA256s (any tampering report is
welcome but belongs upstream).
