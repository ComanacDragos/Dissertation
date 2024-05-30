import json
from collections import Counter

stats = json.load(open("scripts/image_stats.json"))

shapes = [tuple(x['shape']) for x in stats.values()]

print(Counter(shapes))