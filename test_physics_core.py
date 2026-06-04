import time
from rocketpy import Environment, SolidMotor, Rocket, Flight

def run_test():
    print("Initializing Environment...")
    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=0)
    
    # Try to set a standard atmosphere with constant wind
    try:
        env.set_atmospheric_model(type='custom_atmosphere', wind_v=2.0)
    except Exception as e:
        print(f"Could not set custom atmosphere, falling back to standard. Error: {e}")
        env.set_atmospheric_model(type='standard_atmosphere')

    print("Initializing Motor...")
    # Safe, standard amateur motor mock parameters (Generic 20N solid motor)
    motor = SolidMotor(
        thrust_source=[
            [0.0, 20.0],
            [0.5, 25.0],
            [1.0, 15.0],
            [1.5, 0.0]
        ],
        dry_mass=0.1,
        dry_inertia=(0.001, 0.001, 0.0001),
        nozzle_radius=0.015,
        grain_number=4,
        grain_density=1815,
        grain_outer_radius=0.015,
        grain_initial_inner_radius=0.005,
        grain_initial_height=0.05,
        grain_separation=0.001,
        grains_center_of_mass_position=-0.1,
        center_of_dry_mass_position=-0.1,
        nozzle_position=0,
        burn_time=1.5,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    print("Initializing Rocket...")
    rocket = Rocket(
        radius=0.025,
        mass=0.5,
        inertia=(0.01, 0.01, 0.001),
        power_off_drag=0.45,
        power_on_drag=0.45,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    
    rocket.set_rail_buttons(upper_button_position=0.2, lower_button_position=-0.2)
    rocket.add_motor(motor, position=-0.2)
    rocket.add_nose(length=0.1, kind="vonkarman", position=0.5)
    rocket.add_trapezoidal_fins(
        n=4, root_chord=0.05, tip_chord=0.02, span=0.04, position=-0.2, sweep_length=0.01
    )
    rocket.add_tail(top_radius=0.025, bottom_radius=0.02, length=0.04, position=-0.3)

    rocket.add_parachute(
        "Main",
        cd_s=1.5 * 0.3,
        trigger="apogee",
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5)
    )

    print("Running Flight Simulation...")
    start_time = time.time()
    
    flight = Flight(
        rocket=rocket, 
        environment=env, 
        rail_length=2.0, 
        inclination=85, 
        heading=0
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n--- Simulation Results ---")
    print(f"Execution Time: {execution_time:.4f} seconds")
    print(f"Apogee Altitude: {flight.apogee:.2f} meters")
    print(f"Total Flight Time: {flight.t_final:.2f} seconds")

if __name__ == '__main__':
    run_test()
