iis_text = """
      - Path [2, 8, 7, 4] <= 1.44942 * Alt [2, 5, 4]
      - Path [3, 0, 8, 5] <= 1.44942 * Alt [3, 2, 5]
      - Path [3, 0, 8, 5] <= 1.44942 * Alt [3, 6, 5]
      - Path [4, 1, 0, 6] <= 1.44942 * Alt [4, 3, 6]
      - Path [4, 1, 0, 6] <= 1.44942 * Alt [4, 7, 6]
      - Path [5, 2, 1, 7] <= 1.44942 * Alt [5, 8, 7]
      - Path [6, 3, 2, 8] <= 1.44942 * Alt [6, 0, 8]
      - Path [6, 3, 2, 8] <= 1.44942 * Alt [6, 5, 8]
      - Path [0, 8, 5, 4] <= 1.44942 * Alt [0, 1, 4]
      - Path [0, 8, 5, 4] <= 1.44942 * Alt [0, 3, 4]
      - Path [1, 2, 5, 6] <= 1.44942 * Alt [1, 0, 6]
      - Path [1, 2, 5, 6] <= 1.44942 * Alt [1, 7, 6]
      - Path [2, 3, 6, 7] <= 1.44942 * Alt [2, 1, 7]
      - Path [2, 3, 6, 7] <= 1.44942 * Alt [2, 1, 4, 7]
      - Path [2, 3, 6, 7] <= 1.44942 * Alt [2, 8, 7]
      - Path [3, 4, 7, 8] <= 1.44942 * Alt [3, 0, 8]
      - Path [3, 4, 7, 8] <= 1.44942 * Alt [3, 2, 8]
      - Path [3, 4, 7, 8] <= 1.44942 * Alt [3, 6, 0, 8]
"""

import re
from collections import defaultdict

left_edges = defaultdict(int)
right_edges = defaultdict(int)

def path_to_edges(path_str):
    nodes = [int(n) for n in re.findall(r'\d+', path_str)]
    edges = [tuple(sorted((nodes[i], nodes[i+1]))) for i in range(len(nodes)-1)]
    return edges

for line in iis_text.strip().split('\n'):
    parts = line.split('<=')
    left_path = parts[0].split('Path ')[1]
    right_path = parts[1].split('Alt ')[1]
    
    for edge in path_to_edges(left_path):
        left_edges[edge] += 1
    for edge in path_to_edges(right_path):
        right_edges[edge] += 1

print(f"{'Edge':<10} | {'Left Count (Preferred)':<25} | {'Right Count (Alt)':<20} | {'Net Force (Left - Right)'}")
print("-" * 85)

all_edges = set(left_edges.keys()).union(set(right_edges.keys()))
total_left = 0
total_right = 0

for edge in sorted(all_edges):
    l_count = left_edges[edge]
    r_count = right_edges[edge]
    total_left += l_count
    total_right += r_count
    net = l_count - r_count
    print(f"{str(edge):<10} | {l_count:<25} | {r_count:<20} | {net}")

print("-" * 85)
print(f"{'TOTAL':<10} | {total_left:<25} | {total_right:<20} | {total_left - total_right} (Massive Contradiction!)")