## 2024-05-15 - UI Execution Button States
**Learning:** PySide6 reactive interfaces with complex background processing states require strict binding to the state properties (e.g. `is_calculating` or `is_ready_to_run`). When the interface implements an enable/disable pattern for standard run buttons, it's easy to miss secondary controls (like a Phase 1 button or Stop button) leaving them detached from the state and vulnerable to duplicate interactions or confusing states.
**Action:** When implementing or fixing UI buttons tied to asynchronous simulations or processes, trace the `setEnabled` logic to ensure *all* related start/stop inputs react correctly to the system's "idle vs running" and "ready vs not ready" boundaries.
## 2026-05-16 - Adding Tooltips and Shortcuts to Primary Buttons
**Learning:** In PySide6 applications with complex interfaces, adding `setShortcut()` and `setToolTip()` to primary action buttons (like Run/Stop) and improving empty state labels with actionable instructions significantly enhances discoverability and keyboard accessibility without changing the visual layout.
**Action:** Next time, always check if main execution buttons have keyboard shortcuts assigned, and look for 'dead end' empty states that can be rewritten to guide the user towards the next action.
## 2024-05-17 - Button Tooltips and Shortcuts
**Learning:** PySide6 UI elements inside modal dialogs (like `QDialog`) often hide features from users unless explicitly documented. Adding tooltips using `setToolTip` to secondary actions (like 'Load JSON' or 'Clear Curve') clarifies function without adding visual clutter. Associating a keyboard shortcut (like 'Esc') to close operations using `setShortcut` massively improves keyboard navigation inside modals.
**Action:** When inspecting modal dialog UI patterns, check whether utility buttons have tooltips to explain their side effects, and verify keyboard shortcuts exist for modal dismissal.
## 2026-05-26 - [Avoid Tooltips in High-Glare Environments]
**Learning:** In outdoor, high-glare environments (like launch sites for rocket software), tooltips are completely ineffective and should be avoided.
**Action:** Prioritize strong visual feedback (like red borders for validation errors) and robust keyboard navigation (using `setTabOrder`) to guide the user without relying on hover interactions.
## 2024-05-28 - Restoring QSS Focus Indicators
**Learning:** In PySide6, applying custom stylesheets (QSS) to interactive widgets often overrides or removes default OS focus rings.
**Action:** Always explicitly implement `:focus` pseudo-class styles (e.g. `outline: none; border-color: ...`) for keyboard accessibility. If a widget originally lacked a border (`border: none`), use a transparent border and adjust padding proportionally to prevent layout shifts when the focus ring appears.
