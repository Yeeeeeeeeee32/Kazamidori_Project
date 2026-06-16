## 2024-06-16 - Qt Custom Stylesheets Breaking Keyboard Accessibility
**Learning:** In PySide6/Qt applications, when applying custom stylesheets (QSS) to widgets like `QPushButton` or `QToolBox::tab`, default OS/Qt focus indicators are overridden or lost. This severely impacts keyboard accessibility.
**Action:** Always explicitly re-implement `:focus` pseudo-class styles (e.g., `outline: none; border-color: #7eb3ff;`) for clear visual feedback during tab navigation to maintain accessibility.
