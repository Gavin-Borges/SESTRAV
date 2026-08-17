# Bus Factor & Continuity Plan

The "bus factor" is the number of people who would have to suddenly leave the
project before it could no longer be maintained. This document records SESTRAV's
bus factor, its continuity measures, and the status of the backup-maintainer role.
It supports the OpenSSF criteria `access_continuity` (Silver) and `bus_factor`
(Silver SHOULD / Gold MUST).

## Current status

- **Lead maintainer:** Gavin Borges (@Gavin-Borges) - primary author, holds repo
  admin, performs releases and security response.
- **Backup maintainer:** **none, and none is planned** (settled 2026-08-17). This is
  no longer tracked as an action item: the project is solo-maintained and expects to
  remain so, which is why OpenSSF Silver and Gold are formally declined
  (`ROADMAP.md`). The project remains *open* to a qualified co-maintainer appearing
  unprompted - `MAINTAINERS.md` documents the process, and the requirements below
  still apply if that ever happens - but none is being sought. The backup would need
  to be a trusted
  person with the ability and willingness to take over if the lead becomes
  unavailable. They do **not** need to be a daily committer,
  but they MUST have (a) repository **Admin/Maintain** access and (b) a verified
  ability to build, test, and cut a release (see checklist below). Release-artifact
  signing is keyless (Sigstore via GitHub OIDC), so no artifact-signing key changes
  hands; the continuity requirement there is workflow access, not key custody.
  Commit signing is separate and does use a maintainer-held SSH key; tag signing
  is configured to use that same key, but no signed tag has been cut yet
  (see `docs/releasing.md`).
- **Honest current bus factor:** **1**, and expected to stay 1. Do not claim a bus
  factor of 2 on the OpenSSF questionnaire unless it becomes genuinely true.

> **Bus factor 2 is not a goal this project is working toward.** The path is recorded
> here for completeness only: designate a backup, grant them Maintain/Admin on GitHub,
> have them complete the checklist below, then update this file and the OpenSSF
> `bus_factor` answer. Since no candidate exists and none is being sought, the
> project's actual continuity strategy is the two measures under "Continuity measures"
> - archival (Zenodo DOI) and organization ownership - **neither of which requires a
> second person.**

## Continuity measures already in place

Even at bus factor 1, the project is recoverable by others:

- **Public, forkable source:** the full Git history is public on GitHub and the
  project is MIT-licensed - anyone may fork and continue it.
- **Documented, automated build/test/release:** `README.md`, `CONTRIBUTING.md`,
  and the CI workflows fully describe how to build, test, and release without
  tacit knowledge.
- **External archival (PLANNED, not yet done):** mirror tagged releases to an archival
  DOI service (Zenodo via the GitHub integration) so artifacts survive account loss.
  **Status 2026-08-17: not in place.** No DOI is minted - `CITATION.cff` has no
  `identifiers:` block and the deposition docs still carry `zenodo.XXXXXXX`
  placeholders. For a permanently solo-maintained project this is the single most
  important continuity control, because it is the only one that survives the loss of
  the maintainer's GitHub account entirely.
- **No personal-account lock-in (PLANNED, decided 2026-08-17):** move the repository
  to a GitHub **organization** so ownership is transferable without handing over a
  personal account. An organization may have exactly one member, so **this does not
  require a second maintainer** - it is the highest-leverage bus-factor mitigation
  available to a solo project, and it is now a decision rather than a suggestion.
  **Two ordering constraints, both real:**
  1. **Do it BEFORE the first PyPI publish.** The pending Trusted Publisher is bound
     to owner `Gavin-Borges`; changing the owner invalidates that binding and forces
     re-registration against the new owner. See `docs/releasing.md`.
  2. **Budget for the URL sweep.** 172 occurrences of `Gavin-Borges/` across 18
     tracked files (README badges, `CITATION.cff`, docs, workflows) need updating
     after the move. GitHub redirects the old path, so nothing breaks immediately,
     which is exactly why this is easy to leave half-done.

> **These two are the whole realistic bus-factor programme for this project.**
> Recruiting a co-maintainer is not planned (`ROADMAP.md`), so continuity has to come
> from making the work survivable without one: archive it so it outlives the account,
> and make ownership transferable. Both are unstarted as of 2026-08-17.

## Backup-maintainer onboarding checklist

The designated backup confirms they can independently:

- [ ] Clone the repo and create the environment (`conda env create -f environment.yml`
      or `pip install -e .[dev]`).
- [ ] Run the full test suite (`python -m pytest tests/ -v`) to green.
- [ ] Run the Snakemake dry-run (`snakemake --snakefile pipeline.smk --dry-run --cores 1`).
- [ ] Produce a release bundle (`python -m src.release_bundle --output-dir release_artifacts`).
- [ ] Hold repository **Maintain/Admin** access.
- [ ] Push a version tag (`v*`) to trigger `.github/workflows/release.yml` - release
      artifacts are signed keylessly via Sigstore/OIDC, so no key custody is required.

## Succession

**There is no succession plan, and no backup maintainer - neither designated nor a
candidate.** This is a settled position as of 2026-08-17, not a gap awaiting action:
the project is solo-maintained and expects to remain so.

If the lead maintainer becomes unavailable, **the project simply stops receiving
updates.** No transfer of the GitHub repository, the OpenSSF badge entry, the PyPI
project, the signing identity or the archival DOI is arranged, and none of it is
arrangeable in advance by one person acting alone.

The only continuity guarantee is the one in "Continuity measures" above, and it is a
real one: the source, the full history and the build procedure are public and
MIT-licensed, so **any third party may fork the project and continue it** without
needing permission, credentials, or anything held only by the maintainer.

> **An earlier version of this section described a 30-day trigger after which "the
> backup maintainer assumes the lead role."** That was written in the operative present
> about a person who does not exist, in the one section a reader consults to answer
> "what happens if he stops?" - while `:25` of this same file correctly said the bus
> factor is 1. Corrected 2026-08-17.
