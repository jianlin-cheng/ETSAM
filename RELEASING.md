# Releasing ETSAM

A release is made by pushing a version tag. Everything else is automated by
[release.yml](.github/workflows/release.yml), and every artifact gets the tag's version
(tag `v1.2.3` → version `1.2.3`). The GitHub release is created only if everything builds:

1. **Build** (in parallel, with the tag's version):
   - [installer.yml](.github/workflows/installer.yml): the ETSAM installers
     (`ETSAM-<version>-Linux-x86_64.sh`, `ETSAM-<version>-Windows-x86_64.exe`), each test-installed on a CI machine;
   - [napari-plugin.yml](.github/workflows/napari-plugin.yml): the napari plugin
     (`external_plugins/napari-etsam`), as an sdist and a wheel;
   - [chimerax-plugin.yml](.github/workflows/chimerax-plugin.yml): the ChimeraX bundle
     (`external_plugins/chimerax-etsam`), `chimerax_etsam-<version>-py3-none-any.whl`.
2. **Release**, only if all builds succeed: creates the GitHub release with generated release notes,
   with the installers and the ChimeraX bundle attached.
3. **Version commit**, after the release: a commit by `github-actions[bot]` sets the version in
   `pyproject.toml` and `external_plugins/chimerax-etsam/pyproject.toml` on `main` to the tag's version
   (skipped if the tag is not on `main`). Pull it before your next commit. It needs `main` to accept
   pushes from the workflow's token (no branch protection that blocks it).
4. **Publish**, after the release: the napari plugin to [PyPI](https://pypi.org/project/napari-etsam/)
   (a new project the first time, a new version afterwards), and the documentation
   ([deploy-docs.yml](.github/workflows/deploy-docs.yml), run on `main`) to GitHub Pages. The docs are
   deployed only by a release, or when you run that workflow by hand from the **Actions** tab, never by
   a push, so documentation for unreleased changes is not published early. (Temporary exception: while
   **Settings ▸ Pages ▸ Source** is *Deploy from a branch*, GitHub's own `pages-build-deployment`
   publishes `main` on every push, so the docs workflow runs again after it to keep the docs online. Once
   the source is *GitHub Actions*, remove the `workflow_run` trigger from `deploy-docs.yml`.)

If a build fails, nothing is released or published. The three build workflows also run on their own
for pushes to `main` that change them (the installers: from the **Actions** tab), to catch problems
before a release.

The installer and the ChimeraX bundle get the version from the tag through
[`.github/scripts/set_version.py`](.github/scripts/set_version.py), which the workflows run before
building; the napari plugin gets it from the tag through setuptools-scm. The versions in the
`pyproject.toml` files are only used for local builds; the release workflow updates them on `main` after a
successful release, so you never edit them by hand.

## Making a release

1. Make sure `main` is ready.
2. Tag the commit and push the tag:

   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

   Do not create the release on GitHub yourself: the workflow creates it.
3. Follow the **Release** workflow in the **Actions** tab; the installers take the longest, about 20
   minutes. When it finishes, edit the release notes on the release page if you like.
4. Upload the ChimeraX bundle to the Toolshed by hand (it has no upload API): download the `.whl` from
   the release, sign in at <https://cxtoolshed.rbvi.ucsf.edu> and choose **Submit a Bundle**.

If the workflow fails, fix the problem, then move the tag to the fixed commit and push it again:

```bash
git tag -d v1.2.0 && git push origin :refs/tags/v1.2.0   # delete the tag locally and on GitHub
git tag v1.2.0 && git push origin v1.2.0
```

If it failed *after* the release was created (publishing to PyPI or the docs), the release and its
files are already public: re-run the failed jobs from the **Actions** tab instead.

## One-time setup

- **PyPI trusted publishing** (before the first release): on
  <https://pypi.org/manage/account/publishing/>, add a *pending publisher* with PyPI project name
  `napari-etsam`, owner `jianlin-cheng`, repository `ETSAM`, workflow `release.yml` and
  environment `pypi`. No API token is needed. The first release then creates the project on PyPI.
- **GitHub environment** `pypi` (**Settings ▸ Environments**): created automatically on first use;
  add required reviewers there if you want to approve each PyPI upload.
- **GitHub Pages**: **Settings ▸ Pages ▸ Source** must be *GitHub Actions* (already used by the docs).
  Pages only accepts deployments from `main`, so the release workflow does not deploy the docs itself:
  it runs the docs workflow on `main`. Tag releases on `main`, so that the deployed docs match.

## Building the ChimeraX bundle without ChimeraX

The bundle is pure Python, so CI builds it with ChimeraX's bundle builder alone, without installing
ChimeraX. The builder (`ChimeraX-BundleBuilder`) is not on PyPI; the workflow installs it with pip from
ChimeraX's source repository, at the ChimeraX release set in `CHIMERAX_REF`, and builds with
`python -m build --no-isolation --wheel`. The result is the same as ChimeraX's `devel build`. The same
works locally:

```bash
pip install build setuptools Cython numpy packaging wheel \
    "ChimeraX-BundleBuilder @ git+https://github.com/RBVI/ChimeraX@v1.11.1#subdirectory=src/bundles/bundle_builder"
cd external_plugins/chimerax-etsam && python -m build --no-isolation --wheel
```

CI checks the wheel's files and the tool and commands in its metadata, but cannot run ChimeraX: before a
release, test the bundle in ChimeraX (`toolshed install dist/chimerax_etsam-<version>-py3-none-any.whl`).
