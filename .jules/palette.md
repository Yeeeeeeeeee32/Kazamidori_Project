## 2026-06-13 - Native Focus Rings Missing in Qt QSS
**Learning:** When applying custom stylesheets (QSS) to widgets like `QPushButton` or `QToolBox::tab` in PySide6/Qt applications, the default OS/Qt focus indicators are often overridden or lost. This severely impacts keyboard accessibility.
**Action:** Always explicitly re-implement `:focus` pseudo-class styles (e.g., `outline: none; border-color: ...`) for clear visual feedback during tab navigation, and ensure widgets have transparent borders by default if necessary to avoid layout shifts when the focus border is applied.
