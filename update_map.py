import re

with open('ui_qt/map_view.py', 'r') as f:
    content = f.read()

# Add imports
if 'import math' not in content:
    content = 'import math\n' + content

# Update __init__
init_str = """        self._build_ui()
        self._draw_static_items(getattr(app_state, 'target_radius', 0.0))"""

new_init_str = """        self._build_ui()

        # Map interaction state
        self._drag_start = None
        self._is_dragging = False
        self._ghost_marker = None

        # Connect Matplotlib events
        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion_notify)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)

        self._draw_static_items(getattr(app_state, 'target_radius', 0.0))"""

content = content.replace(init_str, new_init_str)

with open('ui_qt/map_view.py', 'w') as f:
    f.write(content)
