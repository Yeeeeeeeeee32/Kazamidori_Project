import re

files_to_check = [
    'core/constants.py', 'core/monte_carlo.py', 'core/wind_model.py', 'core/optimization.py',
    'utils/map_downloader.py', 'utils/geo_math.py', 'ui_qt/sim_controller.py', 'ui_qt/app_state.py'
]

print("Running baseline audit script...")

with open("audit_report.md", "w") as report:
    report.write("# Baseline Audit Report\n\n")

    # Check WGS84 constants duplication
    report.write("## 1. Duplicated Constants (DRY Violation)\n")
    for f in files_to_check:
        with open(f, "r") as file:
            content = file.read()
            if "6371" in content or "6378" in content:
                report.write(f"- Found Earth radius hardcoded in `{f}`\n")

    # Check degree conversion in trigonometric functions
    report.write("\n## 2. Missing Unit Conversions\n")
    for f in files_to_check:
        with open(f, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                # Search for math.cos/sin without math.radians explicitly (naive approach, but good enough for baseline)
                if "math.cos(" in line and "math.radians" not in line and "angle_rad" not in line and "rad)" not in line and "(rad" not in line and "t)" not in line and "phi)" not in line and "angle)" not in line and "dir_rot" not in line and "(ang)" not in line:
                     report.write(f"- Potential missing conversion in `{f}` line {i+1}: `{line.strip()}`\n")
                if "math.sin(" in line and "math.radians" not in line and "angle_rad" not in line and "rad)" not in line and "(rad" not in line and "t)" not in line and "phi)" not in line and "angle)" not in line and "dir_rot" not in line and "(ang)" not in line:
                     report.write(f"- Potential missing conversion in `{f}` line {i+1}: `{line.strip()}`\n")

    # Check explicit naming conventions
    report.write("\n## 3. Ambiguous Variable Naming\n")
    for f in files_to_check:
        with open(f, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                 if re.search(r'\bangle\s*=', line) or re.search(r'\bdir\s*=', line) or re.search(r'\bvelocity\s*=', line) or re.search(r'\bspeed\s*=', line):
                     if not ("_deg" in line or "_rad" in line or "_mps" in line):
                         report.write(f"- Missing unit suffix in `{f}` line {i+1}: `{line.strip()}`\n")

print("Done generating report.")
