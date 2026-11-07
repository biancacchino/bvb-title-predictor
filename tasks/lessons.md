# Lessons

- Use full descriptive variable names in code skeletons (home_rating, rating_change), not math letters (e, s, r). Readers learn from the names.
- Stacked PRs: `gh pr merge --delete-branch` on the bottom PR closes the PR on top for good, because its base branch is gone. Retarget the top PR to `main` before merging the bottom one.
