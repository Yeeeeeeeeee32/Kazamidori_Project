import sys
from unittest.mock import MagicMock
sys.modules['rocketpy'] = MagicMock()
import core.simulation as sim

thrust_curve = [
    (-0.1, 10.0),
    (0.5, 20.0),
    (1.0, 5.0)
]
try:
    motor = sim.build_motor_from_curve(thrust_curve, 1.0, 0.1, 0.05, isp_s=80, grain_density=1700)
    print("OK")
except Exception as e:
    print(f"Error: {e}")
