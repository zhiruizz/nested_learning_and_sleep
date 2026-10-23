# Candidate scope and security review

File-level allowlist export; independent new Git history. Only reviewed text
code, tests, synthetic examples, configs, documentation and aggregate numeric
tables are included. No images, notebooks, HTML outputs, external implementation
trees, raw records, labels, model/adapter/optimizer weights or reversible sequences
are distributed. No Git LFS, submodule, symbolic link, junction or private directory
import is used. Runtime dependencies are installed normally.

Automated content/path/credential and file-type checks were followed by manual
reading of the selected source modules, tests, documents and file list. Staging,
candidate files, all new Git objects and final ZIP are checked. Approved commit
author metadata is the only intentionally included personal contact metadata in
Git; it is absent from the ZIP. Detailed classifications remain outside this package.

These checks reduce disclosure risk; they are not proof of universal absence.
Asset-specific licenses and attribution are recorded in LICENSE.md. Publication
of this reviewed export is authorized; author consultation remains a local draft,
and no author has been contacted.
