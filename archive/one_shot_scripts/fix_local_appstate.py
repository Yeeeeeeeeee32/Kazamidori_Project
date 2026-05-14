import re

with open("ui_qt/app_window.py", "r") as f:
    content = f.read()

local_appstate_insert = """
    # ── View Toggles ───────────────────────────────────────────────────────────
    @property
    def show_kde(self) -> bool: return self._show_kde
    @show_kde.setter
    def show_kde(self, v: bool) -> None:
        if self._show_kde != v:
            self._show_kde = v
            self.needs_redraw.emit()

    @property
    def show_cep(self) -> bool: return self._show_cep
    @show_cep.setter
    def show_cep(self, v: bool) -> None:
        if self._show_cep != v:
            self._show_cep = v
            self.needs_redraw.emit()

    @property
    def show_scatter(self) -> bool: return self._show_scatter
    @show_scatter.setter
    def show_scatter(self, v: bool) -> None:
        if self._show_scatter != v:
            self._show_scatter = v
            self.needs_redraw.emit()

    @property
    def show_burnout(self) -> bool: return self._show_burnout
    @show_burnout.setter
    def show_burnout(self, v: bool) -> None:
        if self._show_burnout != v:
            self._show_burnout = v
            self.needs_redraw.emit()

    @property
    def show_apogee(self) -> bool: return self._show_apogee
    @show_apogee.setter
    def show_apogee(self, v: bool) -> None:
        if self._show_apogee != v:
            self._show_apogee = v
            self.needs_redraw.emit()
"""

# add initializers in AppState inside app_window.py
content = content.replace("        self._wind_history:      list           = []", "        self._wind_history:      list           = []\n        self._show_kde = True\n        self._show_cep = True\n        self._show_scatter = True\n        self._show_burnout = True\n        self._show_apogee = True")

content = content.replace("    @property\n    def wind_speed", local_appstate_insert + "\n    @property\n    def wind_speed")

with open("ui_qt/app_window.py", "w") as f:
    f.write(content)
