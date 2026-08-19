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

1. **Bump the version** in `pyproject.toml` (`[project] version`) to match the tag
   you are about to create (e.g. `2.0.2`). The build names artifacts from this
   field, so it must match the tag. The release workflow **enforces** this with a
   fail-fast "Verify tag matches package version" step, so a mismatch aborts the
   release before any artifact is built. Commit it:

   ```bash
   git commit -am "release: v2.0.2"
   ```

2. **Create a signed, annotated tag** and push it:

   ```bash
   git tag -s v2.0.2 -m "SESTRAV v2.0.2"
   git push origin v2.0.2
   ```

3. The **Release workflow** runs automatically and:
   - builds `dist/*.tar.gz` + `dist/*.whl`,
   - generates `SHA256SUMS.txt`,
   - builds a checksummed results bundle (`src/release_bundle.py`: a zip + manifest of the
     tracked canonical `results/*` artifacts, so a reader can verify a release's reported
     numbers against the exact files that produced them),
   - attaches a Sigstore provenance attestation covering the dist and results-bundle files,
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

# Verify the tag signature (only for tags cut with `git tag -s`;
# tags through v2.0.3 are annotated but unsigned):
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
4. Repository variable `PYPI_PUBLISH` gates the publish job. **It is currently `true`**
   (restored 2026-08-17 after the publisher was confirmed; it had been set `false`
   earlier the same day purely as a precaution while the registration was unverified).
   Set it to `false` to disable publishing without touching the workflow.

> **ORDERING CONSTRAINT - read before cutting a tag that publishes.** The pending
> trusted publisher above is bound to **Owner: `Gavin-Borges`**, a personal account.
> Migrating this repository to a GitHub organization (planned - see `BUS_FACTOR.md`)
> **changes the owner and invalidates that binding.** Do the org migration BEFORE the
> first publishing tag. Publishing first is not fatal, but it means re-registering the
> trusted publisher against the new owner on the existing PyPI project afterwards,
> which is more steps and easy to forget.

### How a release publishes to PyPI

After the `release` job completes (build -> attest -> GitHub Release), the `publish`
job is triggered, pauses for reviewer approval, then runs
`pypa/gh-action-pypi-publish` which exchanges a short-lived GitHub OIDC token for
a PyPI upload credential automatically. No static credentials are involved.

## Badge status (as shipped)

The first release (**v2.0.2**) was published via this workflow with a Sigstore
build-provenance attestation, so on the badge form
(<https://www.bestpractices.dev/projects/13191>) `signed_releases` is recorded as
**Met** (cryptographic provenance over the release artifacts, verifiable with
`gh attestation verify`).

`version_tags_signed` remains **Unmet**: every tag through v2.0.3 is annotated but
unsigned, because no personal signing key was configured at the time those
releases were cut. It is a SUGGESTED (not MUST) criterion, so it does not affect
the tier. A maintainer SSH signing key is now configured locally (`gpg.format=ssh`
with `commit.gpgsign` and `tag.gpgsign` enabled), so the criterion is met on the
next release simply by tagging with `git tag -s` and verifying with `git tag -v`.
