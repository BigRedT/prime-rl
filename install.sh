#!/usr/bin/env bash
set -e

eval "$(ssh-agent -s)"
echo "$SSH_KEY" | ssh-add -

uv sync --all-extras
