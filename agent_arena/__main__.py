# Use an absolute import so the same entry point works both with
# ``python -m agent_arena`` and when PyInstaller executes this file as the
# frozen ``__main__`` module.
from agent_arena.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
