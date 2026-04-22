#!/usr/bin/env bash
set -e

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  export UV_PROJECT_ENVIRONMENT=/tmp/venv
fi

# Mounted as a secret while creating beaker session.
# Start an agent only if one isn't already available, then always add the expected key.
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  eval "$(ssh-agent -s)" >/dev/null
fi

if [[ ! -f /root/.ssh/ssh_key ]]; then
  echo "Expected SSH key at /root/.ssh/ssh_key but it was not found. This SSH key is needed to install agent_rl as a dependency." >&2
  exit 1
fi

ssh-add /root/.ssh/ssh_key

uv sync --all-extras
