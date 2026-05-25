import os
import re

for root, dirs, files in os.walk('core'):
    for file in files:
        if not file.endswith('.py'): continue
        filepath = os.path.join(root, file)
        with open(filepath, 'r') as f:
            content = f.read()
            if 'lat' in content.lower() or 'lon' in content.lower():
                print(f"File: {filepath} contains lat/lon")
