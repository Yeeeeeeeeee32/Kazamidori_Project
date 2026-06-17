
## 2024-06-17 - PySide6 Custom Stylesheets Override Focus Indicators
**Learning:** When applying custom stylesheets (QSS) to widgets like `QPushButton` or `QToolBox::tab` in PySide6/Qt, the default OS/Qt focus indicators are overridden or lost. This severely impacts keyboard accessibility.
**Action:** Always explicitly re-implement `:focus` pseudo-class styles (e.g., `outline: none; border-color: ...`) for clear visual feedback during tab navigation. Ensure widgets have transparent borders by default if necessary to avoid layout shifts when the focus border is applied.
