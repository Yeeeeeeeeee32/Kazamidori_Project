with open('ui_qt/map_view.py', 'r') as f:
    content = f.read()

handlers_code = """
    def _on_button_press(self, event):
        if event.inaxes != self.ax: return
        if event.key == 'control' and event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                # Check if click is near origin (0, 0)
                dist = math.hypot(event.xdata, event.ydata)
                # Allowing some leeway to grab the launch site, e.g., within 10% of axis range or a fixed radius
                # We will just assume if control is held, they want to drag it
                self._drag_start = (event.xdata, event.ydata)
                self._is_dragging = True

                # Create a ghost marker
                self._ghost_marker, = self.ax.plot([event.xdata], [event.ydata], marker='*', markersize=15,
                                                  color='white', alpha=0.5, zorder=20)
                self.canvas.draw_idle()

    def _on_motion_notify(self, event):
        if not self._is_dragging: return
        if event.inaxes != self.ax: return
        if event.xdata is not None and event.ydata is not None:
            if self._ghost_marker:
                self._ghost_marker.set_data([event.xdata], [event.ydata])
                self.canvas.draw_idle()

    def _on_button_release(self, event):
        if not self._is_dragging: return
        self._is_dragging = False

        if self._ghost_marker:
            self._ghost_marker.remove()
            self._ghost_marker = None

        if event.xdata is not None and event.ydata is not None and self._drag_start is not None:
            dx = event.xdata - self._drag_start[0]
            dy = event.ydata - self._drag_start[1]

            try:
                current_lat = float(self._state.launch_lat)
                current_lon = float(self._state.launch_lon)

                delta_lat = dy / 111111.0
                delta_lon = dx / (111111.0 * math.cos(math.radians(current_lat)))

                # Update AppState which will trigger UI updates and redraw
                self._state.launch_lat = current_lat + delta_lat
                self._state.launch_lon = current_lon + delta_lon
            except Exception as e:
                print(f"Error updating coordinates: {e}")

        self._drag_start = None
        self.canvas.draw_idle()
"""

# Insert before _on_reset_view
content = content.replace('    def _on_reset_view(self):', handlers_code + '\n    def _on_reset_view(self):')

with open('ui_qt/map_view.py', 'w') as f:
    f.write(content)
