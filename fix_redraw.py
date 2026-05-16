with open('ui_qt/map_view.py', 'r') as f:
    content = f.read()

# Replace lambda _: self._render_result with something that definitely redraws correctly without just passing `{}`
# because if it passes `{}`, it clears the map. Wait, the original code had:
# app_state.launch_lat_changed.connect(lambda _: self._render_result(getattr(self._state, 'simulation_result', {}) or {}))
# This is correct if `simulation_result` is updated.
# Let's write a helper method `_render_current_state` and connect signals to that, which is cleaner.

new_method = """
    def _render_current_state(self):
        result = getattr(self._state, 'simulation_result', {}) or {}
        self._render_result(result)
"""

content = content.replace("    def _build_ui(self):", new_method + "\n    def _build_ui(self):")

# Now update the connections:
import re
content = re.sub(r'lambda _: self._render_result\(getattr\(self._state, \'simulation_result\', {}\) or {}\)',
                 'lambda _: self._render_current_state()', content)

# But we also should connect `launch_lat_changed` and `launch_lon_changed` properly to `_render_current_state`.
# Wait, let's just make sure they call `_render_current_state`.

with open('ui_qt/map_view.py', 'w') as f:
    f.write(content)
