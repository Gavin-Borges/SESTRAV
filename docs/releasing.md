# Releasing SESTRAV (signed, with provenance)

SESTRAV releases are **cryptographically verifiable**. Pushing a version tag runs
`.github/workflows/release.yml`, which builds the Python distribution, produces a
keyless [SLSA build-provenance](https://slsa.dev/) attestation (Sigstore, via
GitHub OIDC — no maintainer-managed keys), and publishes the artifacts plus a
SHA-256 manifest to a GitHub Release.

This is the mechanism behind OpenSSF Best Practices `signed_releases` and
`version_tags_signed`.

## One-time setup: signed git tags (`version_tags_signed`)

Sign tags with your existing GitHub SSH key (no GPG needed):

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub   # your public key
git config --global tag.gpgSign true
```

Then add the **same public key** to GitHub as a *signing* key:
GitHub → Settings → SSH and GPG keys → **New SSH key** → Key type: *Signing Key*.
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
   - attaches a Sigstore provenance attestation,
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

# Verify the signed tag:
git tag -v v2.0.2
```

## Publishing to PyPI (Trusted Publishers — no API token)

The publish job in `release.yml` authenticates to PyPI using **OpenID Connect
Trusted Publishers** — no API token or GitHub secret is required.

### One-time setup (already complete)

1. PyPI account created with 2FA enabled.
2. A **pending trusted publisher** is registered at `pypi.org` → Account settings
   → Publishing with:
   - Owner: `Gavin-Borges`, Repository: `SESTRAV`
   - Workflow: `release.yml`, Environment: `pypi`
3. GitHub environment `pypi` configured with **Required reviewers** — every publish
   attempt pauses for manual approval before proceeding.
4. Repository variable `PYPI_PUBLISH=true` gates the publish job; set it to `false`
   to disable publishing without touching the workflow.

### How a release publishes to PyPI

After the `release` job completes (build → attest → GitHub Release), the `publish`
job is triggered, pauses for reviewer approval, then runs
`pypa/gh-action-pypi-publish` which exchanges a short-lived GitHub OIDC token for
a PyPI upload credential automatically. No static credentials are involved.

## Badge status (as shipped)

The first release (**v2.0.2**) was published via this workflow with a Sigstore
build-provenance attestation, so on the badge form
(<https://www.bestpractices.dev/projects/13191>) `signed_releases` is recorded as
**Met** (cryptographic provenance over the release artifacts, verifiable with
`gh attestation verify`).

`version_tags_signed` remains **Unmet**: v2.0.2 used an annotated but unsigned git
tag because no personal signing key was configured at release time. It is a
SUGGESTED (not MUST) criterion, so it does not affect the tier. To meet it on a
future release, set up an SSH/GPG signing key, tag with `git tag -s`, and verify
with `git tag -v`.
