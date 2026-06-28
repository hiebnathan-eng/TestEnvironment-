# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository (`TestEnvironment-`) is currently empty — it has no source code,
build configuration, or commit history yet. This file is a starting point and
should be expanded as the project takes shape.

When real code lands, update the sections below so future Claude Code sessions
have accurate, actionable context.

## What to fill in as the project grows

### Project overview
- One or two sentences on what this project does and who it's for.
- The primary language(s), framework(s), and runtime versions.

### Setup
- How to install dependencies (e.g. `npm install`, `pip install -r requirements.txt`, `go mod download`).
- Any required environment variables or local config files (and where to copy them from).

### Common commands
Document the canonical commands once they exist, for example:
- **Build:** how to compile/build the project.
- **Run:** how to start the app locally.
- **Test:** how to run the full test suite and a single test.
- **Lint/format:** how to check and auto-fix style.

### Architecture
- The high-level layout: top-level directories and what each is responsible for.
- Key modules, services, or entry points and how they fit together.
- Anything non-obvious about data flow, state, or external integrations that a
  newcomer couldn't infer by reading a single file.

### Conventions
- Code style, naming, and structural patterns the project follows.
- Branching and commit-message conventions.
- Testing expectations for new code.

## Working agreements

- Keep this file accurate. When you change how the project is built, run, or
  tested, update the relevant section here in the same change.
- Prefer documenting the *why* and the non-obvious over restating what is
  already clear from reading the code.
