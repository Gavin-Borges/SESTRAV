# Bus Factor & Continuity Plan

The "bus factor" is the number of people who would have to suddenly leave the
project before it could no longer be maintained. This document records SESTRAV's
bus factor, its continuity measures, and the status of the backup-maintainer role.
It supports the OpenSSF criteria `access_continuity` (Silver) and `bus_factor`
(Silver SHOULD / Gold MUST).

## Current status

- **Lead maintainer:** Gavin Borges (@Gavin-Borges) - primary author, holds repo
  admin, performs releases and security response.
- **Backup maintainer:** none currently designated - **action required.** The
  project is open to a qualified co-maintainer taking on this role; see
  `MAINTAINERS.md` for how to get involved. The backup should be a trusted
  person with the ability and willingness to take over if the lead becomes
  unavailable. They do **not** need to be a daily committer,
  but they MUST have (a) repository **Admin/Maintain** access and (b) a verified
  ability to build, test, and cut a release (see checklist below). Release-artifact
  signing is keyless (Sigstore via GitHub OIDC), so no artifact-signing key changes
  hands; the continuity requirement there is workflow access, not key custody.
  Commit and tag signing is separate and does use a maintainer-held SSH key
  (see `docs/releasing.md`).
- **Honest current bus factor:** **1** until a backup maintainer is designated and
  has completed the onboarding checklist below. Do not claim a bus factor of 2 on
  the OpenSSF questionnaire until this is genuinely true.

> To reach bus factor 2: designate the backup above, grant them Maintain/Admin on
> GitHub, have them complete the checklist, then update this file (and the OpenSSF
> `bus_factor` answer) accordingly.

## Continuity measures already in place

Even at bus factor 1, the project is recoverable by others:

- **Public, forkable source:** the full Git history is public on GitHub and the
  project is MIT-licensed - anyone may fork and continue it.
- **Documented, automated build/test/release:** `README.md`, `CONTRIBUTING.md`,
  and the CI workflows fully describe how to build, test, and release without
  tacit knowledge.
- **External archival (recommended):** mirror tagged releases to an archival DOI
  service (e.g. Zenodo via the GitHub integration) so artifacts survive account loss.
- **No personal-account lock-in (recommended):** consider moving the repository to
  a GitHub **organization** so ownership is not tied to one personal account.

## Backup-maintainer onboarding checklist

The designated backup confirms they can independently:

- [ ] Clone the repo and create the environment (`conda env create -f environment.yml`
      or `pip install -e .[dev]`).
- [ ] Run the full test suite (`python -m pytest tests/ -v`) to green.
- [ ] Run the Snakemake dry-run (`snakemake --snakefile pipeline.smk --dry-run --cores 1`).
- [ ] Produce a release bundle (`python -m src.release_bundle --output-dir release_artifacts`).
- [ ] Hold repository **Maintain/Admin** access.
- [ ] Hold `workflow` permission to run `.github/workflows/release.yml` (release
      artifacts are signed keylessly via Sigstore/OIDC - no key custody required).

## Succession

If the lead maintainer is unavailable for **more than 30 days** without notice,
the backup maintainer assumes the lead role, announces the transition in an Issue,
and updates `GOVERNANCE.md` and this file.
