# Knowledge base sources

Every document in this folder must be listed here with its origin and licence before it is indexed.
Only add material you have the right to use. Do not add content you cannot verify.

## Folder meaning

| Folder | Status meaning | Shown to users as |
|---|---|---|
| `verified/` | Checked ISL reference material (teacher-approved, or from an approved dictionary with permission) | verified |
| `sample/` | Test or demo material written for the pipeline. Not ISL truth. | sample |
| `unverified/` | Material not yet checked by a qualified person | unverified |

## Documents

| File | Folder | Origin | Author | Licence / permission | Status | Date added | Notes |
|---|---|---|---|---|---|---|---|
| `sample/sample_project_info.md` | sample | Written by the SignSync AI project team for pipeline testing | SignSync AI team (Member 4) | Project-internal, contains no third-party text | sample | (fill in) | Contains only general statements; no sign meanings |

## How to add a document

1. Copy the file into `verified/` (only after checking it) or `unverified/`.
2. Add a row above with the origin, licence or written permission, and the date.
3. Run `python -m rag.scripts.build_index` from the backend folder.
| `unverified/isl_basics_reference.md` | unverified | Written by the SignSync AI team as a general introduction to sign language structure. Contains no sign meanings. | SignSync AI team (Member 4) | Project-internal, no third-party text | unverified | (fill in) | Must be checked by a qualified ISL teacher before teaching use |
