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
   field, so it must match the tag. Commit it:

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

## After the first signed release

On the badge form (<https://www.bestpractices.dev/en/projects/13191>) mark
`signed_releases` and `version_tags_signed` as **Met**, citing this workflow and
the verification commands above.
