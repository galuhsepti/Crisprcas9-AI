# Phase 18C Search Safety Policy

## Scope

This operational policy applies to all preflight and execution work for Phase
18C. It hardens repository exploration without changing the locked Phase 18B
scientific protocol, datasets, hashes, models, or experiment matrix.

## Required Inspection Mode

Phase 18C work must use **allowlisted path inspection**. Read only explicit
project, source, test, and approved development-data paths needed for the
current operation. Do not delegate an unrestricted repository-content search.

The following exact repository search exclusions are installed in both
`.rgignore` and `.ignore`:

```text
data/raw/Moreno-Mateos.csv
```

The rule is intentionally narrow. It does not hide other files under `data/`.

## Prohibited Search Behavior

Do not run commands or tools equivalent to any of the following during Phase
18C unless the locked path is explicitly excluded before execution:

```text
rg . data/
grep -R ... data/
Get-ChildItem ... followed by recursive content inspection across data/
generic delegated repository-content search that includes data/raw/
```

When a content search is necessary, restrict it to explicit source paths. If a
broader scope is unavoidable, require an exclusion equivalent to:

```text
--glob '!data/raw/Moreno-Mateos.csv'
```

Tools that do not honor `.rgignore`, `.ignore`, or an equivalent explicit
exclusion must not search recursively across `data/`.

## Runtime Requirement

Future Phase 18C entry points must keep
`src.experiment_protocols.phase18c_access.phase18c_access_guard` active for the
entire process. The guard permits ordinary training operations but rejects a
locked-name path before resolution and rejects resolved aliases before open.

Guard tests must use temporary nonexistent or synthetic paths only. The real
locked CSV must never be opened, read, or hashed to test the guard.

## Incident Boundary

The aborted Phase 18C preflight attempt is recorded in `docs/decisions.md`. It
produced no modeling result, training run, model artifact, prediction artifact,
or scientific Phase 18C classification.
