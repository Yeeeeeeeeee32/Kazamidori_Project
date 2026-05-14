import re

with open("ui_qt/app_state.py", "r") as f:
    content = f.read()

signals_insert = """
    # ── View Toggles ───────────────────────────────────────────────────────────
    show_kde_changed     = Signal(bool)
    show_cep_changed     = Signal(bool)
    show_scatter_changed = Signal(bool)
    show_burnout_changed = Signal(bool)
    show_apogee_changed  = Signal(bool)
"""

content = content.replace("    cep_probability_changed = Signal(float)\n", "    cep_probability_changed = Signal(float)\n" + signals_insert)

init_insert = """
        # View toggles
        self._show_kde     = True
        self._show_cep     = True
        self._show_scatter = True
        self._show_burnout = True
        self._show_apogee  = True
"""

content = content.replace("        self._phase2_active = False\n", "        self._phase2_active = False\n" + init_insert)

properties_insert = """
    # ── View Toggles ───────────────────────────────────────────────────────────

    @Property(bool, notify=show_kde_changed)
    def show_kde(self) -> bool:
        return self._show_kde

    @show_kde.setter
    def show_kde(self, value: bool) -> None:
        if self._show_kde != value:
            self._show_kde = value
            self.show_kde_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_cep_changed)
    def show_cep(self) -> bool:
        return self._show_cep

    @show_cep.setter
    def show_cep(self, value: bool) -> None:
        if self._show_cep != value:
            self._show_cep = value
            self.show_cep_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_scatter_changed)
    def show_scatter(self) -> bool:
        return self._show_scatter

    @show_scatter.setter
    def show_scatter(self, value: bool) -> None:
        if self._show_scatter != value:
            self._show_scatter = value
            self.show_scatter_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_burnout_changed)
    def show_burnout(self) -> bool:
        return self._show_burnout

    @show_burnout.setter
    def show_burnout(self, value: bool) -> None:
        if self._show_burnout != value:
            self._show_burnout = value
            self.show_burnout_changed.emit(value)
            self.needs_redraw.emit()

    @Property(bool, notify=show_apogee_changed)
    def show_apogee(self) -> bool:
        return self._show_apogee

    @show_apogee.setter
    def show_apogee(self, value: bool) -> None:
        if self._show_apogee != value:
            self._show_apogee = value
            self.show_apogee_changed.emit(value)
            self.needs_redraw.emit()

"""

content = content.replace("    # ── Overlay display parameters ─────────────────────────────────────────────\n", properties_insert + "    # ── Overlay display parameters ─────────────────────────────────────────────\n")

with open("ui_qt/app_state.py", "w") as f:
    f.write(content)
