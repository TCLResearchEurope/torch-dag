#!/bin/bash
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

version=v`cat VERSION`

git pull

if [ $(git tag -l ${version}) ]; then
    printf "${RED}Error:${NC} Tag ${version} already exists. Cannot override an existing version. Update the VERSION file.\n" >&2
    exit 1
fi

# Check if the file is in the working directory (unstaged changes)
if git diff --name-only | grep -q "VERSION"; then
    git add VERSION
fi

# Check if the file is in the git index (staged changes)
if git diff --name-only --cached | grep -q "VERSION"; then
    staged_files_count=$(git diff --cached --name-only | wc -l)

    if [ "$staged_files_count" -eq 1 ]; then
        git commit -m "Update VERSION"
    else
        git reset VERSION
        printf "${RED}Error:${NC} There are $staged_files_count staged, uncommitted files in Git."
        exit 1
    fi
fi

git push
git tag -a ${version} -m "Release ${version}"
git push origin ${version}