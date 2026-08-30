name: example-broken-frontmatter
description: Synthetic example skill with DELIBERATELY BROKEN frontmatter -- there is no opening delimiter, so a skill loader never loads this file and nothing errors anywhere.

# Example: broken frontmatter

This file is missing its opening `---` line. That single defect means the
frontmatter cannot be parsed, so the skill never loads -- silently, with no error
in any log.

It is committed on purpose. An improver that writes a well-reasoned improvement
into an unparseable file has changed nothing, will be told by its own diff that it
changed something, and no runtime signal will contradict it. This repository
therefore reports "does not load" as a first-class finding, and refuses to accept
a proposal that would remove a working frontmatter block.
