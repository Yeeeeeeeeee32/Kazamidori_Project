## 2023-10-27 - PySide6 Focus Accessibility and Layout Shifts
**Learning:** In PySide6 and QSS, adding focus outlines/borders to elements like `QPushButton` or `QToolBox::tab` for keyboard accessibility can cause unexpected layout shifts if the elements did not have borders originally.
**Action:** When adding focus styles, use `border: xpx solid transparent` and adjust padding to compensate on the un-focused state, or use `outline: none; border-color: ...` if the border was already present.
