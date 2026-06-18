## 2024-05-18 - PySide6 Custom Styling Focus Rings
**Learning:** Applying custom QSS to widgets like QPushButton or QToolBox::tab often removes default OS focus indicators, severely hurting keyboard accessibility.
**Action:** Always explicitly implement :focus pseudo-classes (with transparent borders by default if necessary to prevent layout shifts) when adding custom styles to interactive UI elements.
