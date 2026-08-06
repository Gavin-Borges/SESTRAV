# SESTRAV Maintainers

This file records the project's maintainer roster, which the OpenSSF Best Practices
Silver criteria `bus_factor >= 2` and `two_person_review` depend on. Neither criterion
is met today: pull requests from contributors require maintainer review, but with a
single maintainer `two_person_review` is not yet in force (GitHub does not permit
self-approval). See `BUS_FACTOR.md` for the honest current status.

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
  OIDC Trusted Publisher workflow is enabled and is scheduled by any `v*` tag; it
  waits on manual approval under the `pypi` environment's required-reviewer rule.

## Adding a Maintainer

1. Add a row to the table above with name, GitHub username, role, and affiliation.
2. Invite the GitHub user as a repository collaborator with Write access.
3. Update `CODEOWNERS` if path-specific ownership should change.
4. Enable branch protection requiring at least one maintainer review:
   GitHub Settings > Branches > Branch protection rules > main >
   "Require a pull request before merging" > "Required approvals: 1"
