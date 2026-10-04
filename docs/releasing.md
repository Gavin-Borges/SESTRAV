# Releasing SESTRAV (signed, with provenance)

SESTRAV releases are **cryptographically verifiable**. Pushing a version tag runs
`.github/workflows/release.yml`, which builds the Python distribution, produces a
keyless [SLSA build-provenance](https://slsa.dev/) attestation (Sigstore, via
GitHub OIDC - no maintainer-managed keys), and publishes the artifacts plus a
SHA-256 manifest to a GitHub Release.

This is the mechanism behind OpenSSF Best Practices `signed_releases`.
`version_tags_signed` is separate: it depends on signing the tag locally with
`git tag -s`, not on this workflow.

## One-time setup: signed git tags (`version_tags_signed`)

Sign tags with your existing GitHub SSH key (no GPG needed):

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub   # your public key
git config --global tag.gpgSign true
```

Then add the **same public key** to GitHub as a *signing* key:
GitHub -> Settings -> SSH and GPG keys -> **New SSH key** -> Key type: *Signing Key*.
GitHub will then show your tags/commits as **Verified**.

## Cutting a release

1. **Bump the version in all seven carriers, two of which are gated.** The release
   workflow's fail-fast "Verify tag matches package version" step runs before anything
   is built, so a mismatch in either gated file aborts the release with no artifact
   produced. Its name mentions only the package version; it actually reads two files
   and enforces three conditions:

   | File | Field | What the step requires |
   |---|---|---|
   | `pyproject.toml` | `[project] version` | Exactly the tag with its leading `v` stripped (tag `v2.0.2` -> `2.0.2`). The build also names the artifacts from this field. |
   | `CITATION.cff` | top-level `version:` | Present, and the same value. A missing field fails just as hard as a wrong one. |
   | `CITATION.cff` | top-level `date-released:` | Present, parseable as an ISO calendar date (`YYYY-MM-DD`), and not later than the UTC date of the workflow run. |

   `CITATION.cff` is the one that gets forgotten, which is why it is gated: the check
   was added after that file advertised a version and a release date for which no tag
   had ever been pushed.

   A later step in the same job installs the built wheel and asserts that
   `sestrav.__version__` matches the tag. That is not a third file to edit -
   `sestrav/__init__.py` resolves the version from installed package metadata, so it
   reports whatever `pyproject.toml` declared.

   **Five further carriers, in four more files, state the current version and that
   step reads none of them**, so each can go stale through a release without failing
   it. Bump them in the same commit:

   | File | Carrier | The form it takes |
   |---|---|---|
   | `README.md` | version badge | `badge/version-<X.Y.Z>-` inside the shields.io URL |
   | `README.md` | BibTeX entry | `version   = {<X.Y.Z>}` in the BibTeX citation block |
   | `USAGE.md` | CLI version output | the `sestrav version : <X.Y.Z>` line in the recorded `sestrav info` output |
   | `api/main.py` | source-run API fallback | `_APP_VERSION = "<X.Y.Z>"`, the value served when the package metadata is unavailable |
   | `docs/model_cards/rf_31feature_integrated.md` | model-card version field | `- **Version:** SESTRAV v<X.Y.Z>` |

   That is seven current-version carriers across six files: the two the step gates,
   and these five. The other model cards under `docs/model_cards/` are not carriers -
   they record the version of the model they describe (`v2.0`, `v2.1-dev`), not the
   project's, and none of them states an `X.Y.Z` version. `sestrav/__init__.py` is
   still not a carrier either, for the reason above.

   Commit every carrier together:

   ```bash
   git commit -am "release: v2.0.2"
   ```

2. **Create a signed, annotated tag** and push it:

   ```bash
   git tag -s v2.0.2 -m "SESTRAV v2.0.2"
   git push origin v2.0.2
   ```

   **The tag push is the point of no return, and nothing on the server stops it.**
   Ruleset `Protect Main Branch` targets `refs/heads/main` only; no ruleset covers
   `refs/tags/*`, so a tag push needs no pull request, no review and no required
   check. It starts the Release workflow and, for a `vX.Y.Z` tag, the container
   workflow, and if the `PYPI_PUBLISH` repository variable is `true` it queues an
   upload PyPI will never let you replace for that version number.

   `scripts/hooks/pre-push` Check 1b is the local mitigation. It re-runs the Release
   workflow's own version assertions before the push instead of after it, on the files
   in the commit the tag points at, so a `pyproject.toml` or `CITATION.cff` mismatch
   costs a corrected commit rather than a public tag that fails its own release. It is
   a copy of those checks, not the workflow: it can block a few forms the workflow
   accepts, and it takes the first `version = "..."` line of `pyproject.toml` instead
   of parsing TOML. Keep the tag message ASCII: the
   `commit-msg` hook gates commit messages but not tag messages, and the `v2.0.3` tag
   message carries an em-dash, which `CONTRIBUTING.md` bans from commit messages and
   staged files.

3. The **Release workflow** runs automatically and:
   - builds `dist/*.tar.gz` + `dist/*.whl`,
   - generates `SHA256SUMS.txt`,
   - builds a checksummed results bundle (`src/release_bundle.py`: a zip + manifest of the
     tracked canonical `results/*` artifacts, so a reader can verify a release's reported
     numbers against the exact files that produced them),
   - attaches a Sigstore provenance attestation covering the sdist, the wheel and the
     results-bundle ZIP (the bundle's `*.manifest.json` is covered only through the copy
     packed inside the ZIP),
   - creates the GitHub Release with all assets and auto-generated notes.

4. **Update `SECURITY.md`** "Release Integrity & Verification" to record the first
   signed version (and, if you also publish a key fingerprint, record it per
   `BUS_FACTOR.md`).

## Verifying a release (what consumers run)

```bash
# Verify the build provenance came from this repository's CI:
gh attestation verify sestrav-2.0.2-py3-none-any.whl --repo Gavin-Borges/SESTRAV

# Verify the checksum manifest:
sha256sum -c SHA256SUMS.txt

# Verify the tag signature. v2.0.3 IS signed; v2.0.2 and earlier are not.
# Note that `git tag -v` checks your local allowed_signers file, which is a
# different question from whether GitHub shows the tag as Verified:
git tag -v vX.Y.Z
```

## Publishing to PyPI (Trusted Publishers - no API token)

The publish job in `release.yml` authenticates to PyPI using **OpenID Connect
Trusted Publishers** - no API token or GitHub secret is required.

### One-time setup (complete)

> **Confirmed 2026-08-17, and this supersedes the 2026-08-16 retraction.** That
> retraction changed this heading from "(already complete)" to "UNCONFIRMED" on the
> grounds that a pending trusted publisher cannot be verified from any public API -
> only by signing in to pypi.org. **The maintainer has now signed in and confirmed it:
> the pending trusted publisher IS registered.** So the original "already complete"
> claim was substantively true, and the retraction - correct as process at the time,
> since an unverifiable claim should not stand - is withdrawn on evidence.
>
> **Still true and load-bearing:** the package name remains unclaimed
> (`https://pypi.org/pypi/sestrav/json` returns 404), so **nothing has ever been
> published** and the publish path has never executed end-to-end. A *pending* publisher
> is exactly the right configuration for that state; it converts to an ordinary trusted
> publisher on the first successful upload.

1. PyPI account created with 2FA enabled.
2. **CONFIRMED 2026-08-17:** a **pending trusted publisher** is registered at `pypi.org`
   -> Account settings -> Publishing with:
   - Owner: `Gavin-Borges`, Repository: `SESTRAV`
   - Workflow: `release.yml`, Environment: `pypi`
3. GitHub environment `pypi` configured with **Required reviewers** - every publish
   attempt pauses for manual approval before proceeding. So a tag QUEUES a publish for
   approval; it does not publish silently. Note that the sole configured reviewer is the
   maintainer, so this is a deliberate-action prompt rather than independent approval.
4. Repository variable `PYPI_PUBLISH` gates the publish job, via
   `if: ${{ vars.PYPI_PUBLISH == 'true' }}` on that job in `release.yml`.
   **This document deliberately does not record the variable's value.** It is an
   owner-operated switch that gets flipped in both directions, so any value written
   here is a status claim that rots between readings. Read the live one yourself,
   every time, before you push a tag:

   ```bash
   gh variable list          # the PYPI_PUBLISH row
   ```

   - Set to `true`: a version tag schedules the publish job, which then waits on the
     step 3 reviewer approval. Treat this as the irreversible setting, because PyPI
     permanently refuses a re-upload of a version number that has already been
     published, so a bad upload cannot be replaced under the same number.
   - Set to anything that is not `true` (`false` included), or absent from the
     repository altogether: the publish job is never scheduled, and the tag produces
     the GitHub Release with its attestation and checksums and nothing else.

   Change it at Settings -> Secrets and variables -> Actions -> Variables, or with
   `gh variable set PYPI_PUBLISH --body false`. The workflow itself never needs
   editing.

> **ORDERING CONSTRAINT - read before cutting a tag that publishes.** The pending
> trusted publisher above is bound to **Owner: `Gavin-Borges`**, a personal account.
> Migrating this repository to a GitHub organization (planned - see `BUS_FACTOR.md`)
> **changes the owner and invalidates that binding.** Do the org migration BEFORE the
> first publishing tag. Publishing first is not fatal, but it means re-registering the
> trusted publisher against the new owner on the existing PyPI project afterwards,
> which is more steps and easy to forget.

### How a release publishes to PyPI

After the `build` job (build and package-data checks), the `verify` job (the pre-publish
install-and-import gate) and the `release` job (artifact digest check, attest ->
GitHub Release) complete, the `publish` job is triggered, pauses for reviewer
approval, then runs
`pypa/gh-action-pypi-publish` which exchanges a short-lived GitHub OIDC token for
a PyPI upload credential automatically. No static credentials are involved.

## Badge status (as shipped)

The first release (**v2.0.2**) was published via this workflow with a Sigstore
build-provenance attestation over its wheel and sdist, verifiable with
`gh attestation verify`. On the badge form
(<https://www.bestpractices.dev/projects/13191>), `signed_releases` is not recorded
as Met. It is a Silver-level
criterion (the project's badge level is Passing), and the form leaves it
unanswered: read 2026-10-01, the project JSON gives
`"signed_releases_status":"?"`, last updated 2026-06-16, and the Silver page
shows it as Unknown. This paragraph previously said it was "recorded as **Met**";
that is not what the form records.

`version_tags_signed` remains **Unmet**, and the reason is not the one this
paragraph used to give. **Corrected 2026-09-15: v2.0.3 IS signed.** Its tag object
carries an SSH signature block, and the local tag is byte-identical to the one on
origin, so the pushed tag carries it too. v2.0.2 and earlier are genuinely
unsigned. It is a SUGGESTED (not MUST) criterion, so it does not affect the tier.

**Signing the next tag is necessary but NOT sufficient, which is the part that was
missing here.** GitHub reports v2.0.3 as `verified: false, reason: unknown_key`,
meaning no SSH signing key is registered on the account under Settings, SSH and
GPG keys, with key type **Signing Key**. Until that registration happens, a tag
cut with `git tag -s` will still display as Unverified on GitHub and the criterion
stays Unmet no matter how it was signed. Local verification is a separate
question: `git tag -v v2.0.3` reports a good signature but `No principal matched`,
because it was signed with a different key than the one `user.signingkey` now
names, so a tag cut with the current key will verify locally while still showing
Unverified on GitHub until the key is registered. Register the key first, then
tag. No CI step verifies tag signatures, so nothing else will catch this.
