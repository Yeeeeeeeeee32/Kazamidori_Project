from ui.app_window import AppWindow

DEFAULT_CONFIG = {
    "wind_uncertainty":    0.20,
    "thrust_uncertainty":  0.05,
    "allowable_uncertainty": 20.0,
    "landing_prob":        90,
    "mc_n_runs":           200,
}

if __name__ == "__main__":
    app = AppWindow(config=DEFAULT_CONFIG)
    app.mainloop()
