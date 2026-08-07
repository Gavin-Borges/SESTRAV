# SESTRAV Maintainers

This file supports the OpenSSF Best Practices Silver badge criteria for `bus_factor >= 2`
and `two_person_review` (see `BUS_FACTOR.md` for the current honest status). All pull
requests to `main` require at least one review from a maintainer listed here.

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
- Publish PyPI releases via the OIDC Trusted Publisher workflow

## Adding a Maintainer

1. Add a row to the table above with name, GitHub username, role, and affiliation.
2. Invite the GitHub user as a repository collaborator with Write access.
3. Update `CODEOWNERS` if path-specific ownership should change.
4. Enable branch protection requiring at least one maintainer review:
   GitHub Settings > Branches > Branch protection rules > main >
   "Require a pull request before merging" > "Required approvals: 1"
