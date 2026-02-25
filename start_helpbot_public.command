#!/bin/zsh
set -e

TARGET="/Users/moldovancsaba/Projects/reply-amanoba/start_helpbot.command"

if [ ! -f "$TARGET" ]; then
  echo "Launcher not found: $TARGET"
  exit 1
fi

exec "$TARGET"
