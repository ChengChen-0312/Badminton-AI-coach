# Repository Hardening

This repository now ignores local videos and generated artifacts for normal commits.

## What is already fixed

- `archive/` is ignored.
- generated folders (`reports/`, `runs/`, `data/distill/`, `data/mlx_train/`) are ignored.
- temporary/root demo images are ignored.
- backup/history files under `src/vision` are ignored.

## Remaining historical size problem

If large media files were committed in older commits, repository history is still heavy.
To fully clean history, rewrite it once on a maintenance branch.

## History rewrite (recommended once)

1. Create a mirror clone:

```bash
git clone --mirror <YOUR_REPO_URL> badminton-ai-coach.git
cd badminton-ai-coach.git
```

2. Install `git-filter-repo` and remove historical heavy paths:

```bash
git filter-repo \
  --path archive --invert-paths \
  --path data/distill --invert-paths \
  --path data/mlx_train --invert-paths \
  --path runs --invert-paths \
  --path image.png --invert-paths \
  --path image-1.png --invert-paths \
  --path image-2.png --invert-paths \
  --path image-3.png --invert-paths
```

3. Force push rewritten history:

```bash
git push --force --all
git push --force --tags
```

## Important

- History rewrite changes commit hashes.
- Coordinate with collaborators before force push.
- Ask everyone to re-clone after cleanup.

