# Knowledge base folder

Put ISL documents here. Supported formats: `.pdf`, `.txt`, `.md`.

- `verified/`: checked ISL reference material only
- `sample/`: test material (shipped with the project)
- `unverified/`: anything not yet checked

Record every file in `SOURCES.md`. Rebuild the index after any change:

    python -m rag.scripts.build_index
