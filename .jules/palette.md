## 2024-05-24 - Qt Custom Stylesheets Override Default Focus Indicators
**Learning:** In PySide6/Qt applications, applying custom stylesheets (QSS) to interactive widgets like `QPushButton` overrides the default OS focus rings, leading to a complete loss of keyboard navigation visibility.
**Action:** Always explicitly re-implement the `:focus` pseudo-class (e.g., `outline: none; border-color: #...;`) when styling interactive Qt elements to ensure keyboard accessibility is maintained.
