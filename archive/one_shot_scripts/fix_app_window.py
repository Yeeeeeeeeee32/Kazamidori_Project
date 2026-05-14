import re

with open("ui_qt/app_window.py", "r") as f:
    content = f.read()

# The reviewer pointed out we can just use self.state (AppWindow has self.state)
# Let's change the definition and usage
content = content.replace("def _draw_real_result(self, ax, res: dict, s: AppState) -> None:", "def _draw_real_result(self, ax, res: dict) -> None:")
content = content.replace("self._draw_real_result(ax, res, s)", "self._draw_real_result(ax, res)")

# In _draw_real_result, add `s = self.state`
content = content.replace('tx = np.asarray(res.get("trajectory_x", [0.0]), dtype=float)', 's = self.state\n        tx = np.asarray(res.get("trajectory_x", [0.0]), dtype=float)')

with open("ui_qt/app_window.py", "w") as f:
    f.write(content)
