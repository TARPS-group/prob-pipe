# Phase 0 — Load knowledge and memory

Reply to the user before you read the documentation. Before the reply, read
only the memory of step 1 and [phases/1-intake.md](1-intake.md), which holds
the first questions. Search the rest of the corpus when a question needs it.

1. Read person memory, at the location that
   [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#locations) gives, if it exists. If
   the opening message names a project, read that project's memory too.
2. Reply to the user first, in a short paragraph:
   - **Returning user:** greet them by name if you know it. Name one project,
     which is the project whose workspace contains the current directory, or
     else the project they worked on most recently. Say in one sentence where
     it stopped and how many other projects memory holds, and name the others
     only when the user asks. Ask whether to resume that
     project, resume another, or start a new one. Mention the working
     preferences only when the user asks about them. If the opening message names a
     project, resume it directly by the
     resume protocol of [INSTRUCTIONS.md](../INSTRUCTIONS.md#resume-protocol):
     read the project's memory and workspace files, summarize where things
     stand and the agreed next step, and continue at the recorded phase.
   - **New user:** say how the session will go, and ask the first group of
     questions of [Phase 1](1-intake.md).
3. Build the corpus while the user answers. If your agent can run commands
   after its reply in the same turn, build it then, and otherwise build it at
   the start of your next turn. Run the script with `<python>`, so that its
   version check describes the environment where ProbPipe is installed:
   `<python> <assistant>/build_corpus.py --repo <checkout> [<corpus>]`
   The corpus directory defaults to the directory `corpus` under the
   person-memory directory. When `<corpus>/MANIFEST.md` already names the
   commit of the checkout, which `git -C <checkout> rev-parse HEAD` prints, use
   that corpus instead. The script flattens the README, the documentation site, and the
   design reference into tiered part files, with the API reference resolved
   from source to signatures and docstrings. It writes `<corpus>/INDEX.md`
   and `<corpus>/MANIFEST.md`, which records the branch, the commit, and the
   version of the checkout and lists the headings of each part with their line
   numbers. For each warning it prints, tell the user in plain words what it
   means for their analysis and whether they need to act, such as an
   installed `probpipe` that comes from outside the checkout. Keep a warning
   that changes nothing for the analysis out of the reply, such as stale
   version metadata of a `probpipe` that imports from the checkout, and
   record it in project memory by step 6.
4. Read in full `<corpus>/INDEX.md`, each `1-overview` part that the manifest
   lists, and [CAPABILITIES.md](../CAPABILITIES.md). The index lists the
   public exports with their signatures, the inference methods with their
   priorities, and the contents of the tutorials and the user guide. When
   your agent's read returns fewer lines than a file holds, read the file in
   consecutive line ranges, which for a part start at the lines of the
   headings that `<corpus>/MANIFEST.md` lists.
5. Check whether ProbPipe imports in that environment, and from where, with
   `<python> -P -c "import probpipe; print(probpipe.__version__, probpipe.__file__)"`.
   The flag `-P` leaves the current directory off the path, so a directory
   that holds another copy of ProbPipe does not import that copy. When the
   file is outside `<checkout>/probpipe/`, tell the user that the environment
   runs another copy of ProbPipe, whose API can differ from the corpus. The
   version comes from the installed metadata, so the commit of step 6
   identifies the code. Note which optional backends of the installation page
   are installed, since they decide whether you can run and validate code
   later.
6. Once Phase 1 creates the project memory, record in it the corpus directory
   and the commit and version that `<corpus>/MANIFEST.md` names, and record
   `<python>` beside them, with the optional backends of step 5 and each
   warning that step 3 kept out of the reply.

## Reading on demand

- Before you propose or write a ProbPipe call, search the corpus and the
  `probpipe/` source of the checkout for it, and read the section that the
  search finds. A part's headings in `<corpus>/MANIFEST.md` give the line
  range of each section, so one read can return one section.
- Each source in a corpus file starts at a `# FILE: <path>` line, and a
  source that continues from the previous part starts at a line that ends in
  `(continued)`.
- Read the design tier, or `design/` in the checkout, only to resolve a term,
  starting from `design/glossary.md`. The design describes the target state,
  so confirm in the code each API that it names.
