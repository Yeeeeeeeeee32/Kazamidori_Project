## $(date +%Y-%m-%d) - Add Focus Visible Styles for QPushButtons
**Learning:** Qt stylesheets (QSS) do not automatically provide prominent focus rings when custom styling is applied to buttons, which degrades keyboard accessibility. Adding explicit `:focus` states using existing token colors (e.g., hover colors) immediately restores accessibility.
**Action:** When overriding default widget styles in Qt (or any UI framework), always explicitly re-implement `:focus` states.
