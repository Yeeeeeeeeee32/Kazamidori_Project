## 2025-02-12 - PySide6 QSS Focus Indicators Overridden
**Learning:** When applying custom stylesheets (QSS) to PySide6 widgets like `QPushButton` or `QToolBox::tab`, the default OS/Qt focus indicators are often overridden or lost, which breaks keyboard accessibility.
**Action:** Always explicitly re-implement `:focus` pseudo-class styles (e.g. `outline: none; border-color: ...`) to ensure clear visual feedback during tab navigation. Ensure widgets have transparent borders by default if necessary to avoid layout shifts on focus.
