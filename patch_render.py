with open('ui_qt/map_view.py', 'r') as f:
    content = f.read()

# Replace _render_result
import re

render_start = content.find('    def _render_result(self, result):')
reset_start = content.find('    def _on_button_press(self, event):')

new_render = """    def _render_result(self, result):
        self.ax.clear()

        # Setup Axes
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.grid(True, linestyle='--', alpha=0.7, color='#45475a')
        self.ax.set_xlabel("East (m)", color='#cdd6f4')
        self.ax.set_ylabel("North (m)", color='#cdd6f4')
        self.ax.tick_params(colors='#cdd6f4')

        try:
            cur_lat = float(getattr(self._state, 'launch_lat', 0.0))
            cur_lon = float(getattr(self._state, 'launch_lon', 0.0))
            self.ax.set_title(f"Launch Site: {cur_lat:.5f}, {cur_lon:.5f}", color='#cdd6f4')
        except Exception:
            pass

        # To keep track of all points for manual bounds
        all_x = [0.0]
        all_y = [0.0]

        # Launch Site
        self.ax.scatter(0, 0, marker='*', s=150, color='#4488ff', label='Launch Site', zorder=10)

        target_radius = getattr(self._state, 'target_radius', 0.0) or 0.0
        if target_radius > 0:
            target_circle = patches.Circle((0, 0), radius=target_radius, edgecolor='#0055ff', facecolor='none', linestyle='--', zorder=5)
            self.ax.add_patch(target_circle)
            all_x.extend([-target_radius, target_radius])
            all_y.extend([-target_radius, target_radius])

        if not result:
            self.canvas.draw()
            return

        impact_x = float(result.get('impact_x', result.get('land_x', 0.0)))
        impact_y = float(result.get('impact_y', result.get('land_y', 0.0)))
        r90 = float(result.get('r_N_radius', 0.0))
        cep = float(result.get('cep', 0.0))

        scatter_x = result.get('mc_scatter_x', [])
        scatter_y = result.get('mc_scatter_y', [])
        scatter_points = result.get('scatter_points', [])
        if not scatter_x and scatter_points:
            scatter_x = [p[0] for p in scatter_points]
            scatter_y = [p[1] for p in scatter_points]

        ellipse = result.get('cep_ellipse') or result.get('ellipse') # fallback for 'ellipse'
        contours = result.get('kde_contours', [])
        prob = int(result.get('landing_prob', 90))
        apogee = float(result.get('apogee_m', 0.0))
        tof = float(result.get('hang_time', 0.0))

        self._info.setText(
            f"R{prob}: {r90:.1f} m  |  CEP50: {cep:.1f} m  |  "
            f"Apogee: {apogee:.0f} m  |  ToF: {tof:.1f} s"
        )

        # Nominal Landing point
        self.ax.scatter(impact_x, impact_y, marker='o', s=40, color='#ff4444', edgecolor='#cc0000', label='Impact Site', zorder=6)
        all_x.append(impact_x)
        all_y.append(impact_y)

        # Impact Scatter
        if len(scatter_x) > 0 and len(scatter_y) > 0 and getattr(self._state, 'show_scatter', True):
            sx = scatter_x[:500]
            sy = scatter_y[:500]
            self.ax.scatter(sx, sy, c='#ff6633', s=10, alpha=0.5, label='MC Scatter', zorder=2)
            all_x.extend(sx)
            all_y.extend(sy)

        # R90 Circle
        if r90 > 0:
            r90_circle = patches.Circle((impact_x, impact_y), radius=r90, edgecolor='#cc0000', facecolor='none', linewidth=2, zorder=5)
            self.ax.add_patch(r90_circle)
            all_x.extend([impact_x - r90, impact_x + r90])
            all_y.extend([impact_y - r90, impact_y + r90])

        # CEP Circle
        if cep > 0 and getattr(self._state, 'show_cep', True):
            cep_circle = patches.Circle((impact_x, impact_y), radius=cep, edgecolor='#9933cc', facecolor='none', linewidth=1.8, linestyle='--', zorder=5)
            self.ax.add_patch(cep_circle)

        # 90% CEP Ellipse
        if ellipse and getattr(self._state, 'show_cep', True):
            cx = ellipse.get('cx', impact_x)
            cy = ellipse.get('cy', impact_y)
            width = ellipse['a'] * 2
            height = ellipse['b'] * 2
            angle_deg = math.degrees(ellipse['angle_rad'])

            ellipse_patch = patches.Ellipse((cx, cy), width, height, angle=angle_deg,
                                            edgecolor='#00bb00', facecolor='none', linewidth=2, label='90% CEP', zorder=3)
            self.ax.add_patch(ellipse_patch)

            # Approximate ellipse bounds for autoscaling
            a = ellipse['a']
            b = ellipse['b']
            # Max possible extent in x and y is roughly cx +- max(a,b)
            r_max = max(a, b)
            all_x.extend([cx - r_max, cx + r_max])
            all_y.extend([cy - r_max, cy + r_max])

        # KDE Contours
        if contours and getattr(self._state, 'show_kde', True):
            for i, contour in enumerate(contours):
                points = contour['points_m'] if 'points_m' in contour else contour
                if points:
                    poly = patches.Polygon(points, closed=True, edgecolor='#cc5500', facecolor='none', linewidth=1.5, zorder=4, label='KDE Contours' if i == 0 else "")
                    self.ax.add_patch(poly)
                    all_x.extend([p[0] for p in points])
                    all_y.extend([p[1] for p in points])

        # Legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            legend = self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#1e1e2e', edgecolor='#45475a', labelcolor='#cdd6f4')
            legend.set_zorder(20)

        # Manual Axis Bounds
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Avoid singular bounds
            if max_x == min_x:
                max_x += 100
                min_x -= 100
            if max_y == min_y:
                max_y += 100
                min_y -= 100

            dx = max_x - min_x
            dy = max_y - min_y

            # 15% margin
            margin_x = dx * 0.15
            margin_y = dy * 0.15

            self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
            self.ax.set_ylim(min_y - margin_y, max_y + margin_y)

        self.figure.tight_layout()
        self.canvas.draw()
"""

new_content = content[:render_start] + new_render + '\n' + content[reset_start:]

with open('ui_qt/map_view.py', 'w') as f:
    f.write(new_content)
