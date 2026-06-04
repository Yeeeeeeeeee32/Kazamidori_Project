## 2025-02-12 - Prevent recursive cursor resets during bi-directional syncing
**Learning:** Using `textChanged` to update AppState for UI inputs mapped to physical states causes recursive cursor resets while typing, which degrades UX for decimal input.
**Action:** Always map empty numeric inputs to `None` in `AppState` via `editingFinished` rather than `textChanged` to ensure the value is only committed after the user has finished typing, preserving cursor position and preventing formatting jank.
