echo "--- DRY Principle (Don't Repeat Yourself) ---"
echo "Checking R_EARTH duplication:"
grep -rn "R_EARTH" .
echo "Checking math utilities overlap:"
ls -la core/geometry_math.py utils/geo_math.py

echo "--- Physics & Math Validation ---"
echo "Checking magic numbers:"
grep -rn "180\.0" core/ utils/

echo "--- Strict Unit Consistency ---"
echo "Checking math.cos() usage without math.radians() or explicit deg/rad variables:"
grep -rn "math.cos(" core/ utils/ | grep -v "rad"

echo "--- Coordinate System Integrity ---"
echo "Checking for 'lat' usage in core/ where it shouldn't be (except simulation.py):"
grep -rn "\.lat" core/
grep -rn "lat_" core/
