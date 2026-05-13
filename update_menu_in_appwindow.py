import re

with open("ui_qt/app_window.py", "r") as f:
    content = f.read()

# Replace the old actions with the new ones we made
old_view_menu = """        # Checkbox actions for map toggles
        self.action_show_map_cep = QAction("Show CEP Ellipse", self, checkable=True)
        self.action_show_map_cep.setChecked(True)
        self.action_show_map_cep.toggled.connect(self.refresh_visuals)
        self._view_menu.addAction(self.action_show_map_cep)

        self.action_show_map_kde = QAction("Show KDE Contours", self, checkable=True)
        self.action_show_map_kde.setChecked(True)
        self.action_show_map_kde.toggled.connect(self.refresh_visuals)
        self._view_menu.addAction(self.action_show_map_kde)"""

new_view_menu = """        # Checkbox actions for plot toggles
        self.action_show_kde = QAction("KDE Contour", self, checkable=True)
        self.action_show_kde.setChecked(self.state.show_kde)
        self.action_show_kde.toggled.connect(lambda c: setattr(self.state, "show_kde", c))
        self._view_menu.addAction(self.action_show_kde)

        self.action_show_cep = QAction("CEP 90% Ellipse", self, checkable=True)
        self.action_show_cep.setChecked(self.state.show_cep)
        self.action_show_cep.toggled.connect(lambda c: setattr(self.state, "show_cep", c))
        self._view_menu.addAction(self.action_show_cep)

        self.action_show_scatter = QAction("Monte Carlo Scatter", self, checkable=True)
        self.action_show_scatter.setChecked(self.state.show_scatter)
        self.action_show_scatter.toggled.connect(lambda c: setattr(self.state, "show_scatter", c))
        self._view_menu.addAction(self.action_show_scatter)

        self.action_show_burnout = QAction("Motor Burnout Point", self, checkable=True)
        self.action_show_burnout.setChecked(self.state.show_burnout)
        self.action_show_burnout.toggled.connect(lambda c: setattr(self.state, "show_burnout", c))
        self._view_menu.addAction(self.action_show_burnout)

        self.action_show_apogee = QAction("Apogee", self, checkable=True)
        self.action_show_apogee.setChecked(self.state.show_apogee)
        self.action_show_apogee.toggled.connect(lambda c: setattr(self.state, "show_apogee", c))
        self._view_menu.addAction(self.action_show_apogee)"""

content = content.replace(old_view_menu, new_view_menu)

with open("ui_qt/app_window.py", "w") as f:
    f.write(content)
