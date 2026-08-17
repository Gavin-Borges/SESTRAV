# SESTRAV Maintainers

This file records the project's maintainer roster, which the OpenSSF Best Practices
Silver criteria `bus_factor >= 2` and `two_person_review` depend on. **Neither criterion
is met, and neither will be**: pull requests from EXTERNAL contributors require
maintainer approval, but the maintainer's own pull requests - which are the large
majority - are merged without review, so `two_person_review` is not in force and cannot
be with one maintainer. Silver and Gold are formally declined on this basis
(`ROADMAP.md`, 2026-08-17). See `BUS_FACTOR.md` for the honest current status.

## Active Maintainers

| Name | GitHub | Role | Affiliation |
|---|---|---|---|
| Gavin Borges | @Gavin-Borges | Lead maintainer | University of Rhode Island |

SESTRAV currently has a single maintainer. The project is open to a qualified
co-maintainer - see "Adding a Maintainer" below for the process, and `BUS_FACTOR.md`
for why this matters and what the role requires.

## Responsibilities

- Review and approve pull requests to `main`
- Triage security reports (see `SECURITY.md`)
- Cut releases (tag, artifacts, provenance attestation). PyPI publishing via the
  OIDC Trusted Publisher workflow is enabled (`PYPI_PUBLISH=true`) and is scheduled
  by any `v*` tag; it then waits on approval under the `pypi` environment's
  required-reviewer rule - where the sole configured reviewer is the maintainer, so
  that is a deliberate-action prompt, not independent approval. The pending Trusted
  Publisher is registered (confirmed 2026-08-17), but **nothing has been published
  yet** and the publish path has never run end to end. Before cutting a tag that is
  meant to publish, read the ordering constraint in `docs/releasing.md`: the
  publisher is bound to the `Gavin-Borges` personal account and a planned migration
  to a GitHub organization would invalidate it.

## Adding a Maintainer

1. Add a row to the table above with name, GitHub username, role, and affiliation.
2. Invite the GitHub user as a repository collaborator with Write access.
3. Update `CODEOWNERS` if path-specific ownership should change.
4. Enable branch protection requiring at least one maintainer review:
   GitHub Settings > Branches > Branch protection rules > main >
   "Require a pull request before merging" > "Required approvals: 1"
