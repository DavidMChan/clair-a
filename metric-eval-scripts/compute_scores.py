import csv
import itertools
import sys
from collections import defaultdict

import numpy as np

with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Step 1: Group by HITId
hitid_to_assignments = defaultdict(list)
hitid_to_assigments_rephrased = defaultdict(list)
for row in data:
    hitid = row["HITId"]
    if row["Input.is_rephrased"] == "1":
        hitid_to_assigments_rephrased[hitid].append(row)
    else:
        hitid_to_assignments[hitid].append(row)

# Step 2: Compute the means for each group
# Three key fields: Answer.justification_quality	Answer.score_fairness	Answer.score_match
# Fot both groups, compute the means (average, micro-average) of these three fields
hitid_to_means = {}
hitid_to_means_rephrased = {}
for hitid, rows in hitid_to_assignments.items():
    means = {
        "justification_quality": np.mean([int(row["Answer.justification_quality"]) for row in rows]),
        "score_fairness": np.mean([int(row["Answer.score_fairness"]) for row in rows]),
        "score_match": np.mean([int(row["Answer.score_match"]) for row in rows]),
    }
    hitid_to_means[hitid] = means

for hitid, rows in hitid_to_assigments_rephrased.items():
    means = {
        "justification_quality": np.mean([int(row["Answer.justification_quality"]) for row in rows]),
        "score_fairness": np.mean([int(row["Answer.score_fairness"]) for row in rows]),
        "score_match": np.mean([int(row["Answer.score_match"]) for row in rows]),
    }
    hitid_to_means_rephrased[hitid] = means

# Print the means (average, micro-average) of these three fields
print("Means (micro-average) for non-rephrased:")
all_samples = list(itertools.chain(*hitid_to_assignments.values()))
print("justification_quality:", np.mean([int(row["Answer.justification_quality"]) for row in all_samples]))
print("score_fairness:", np.mean([int(row["Answer.score_fairness"]) for row in all_samples]))
print("score_match:", np.mean([int(row["Answer.score_match"]) for row in all_samples]))

print("Means (micro-average) for rephrased:")
all_samples = list(itertools.chain(*hitid_to_assigments_rephrased.values()))
print("justification_quality:", np.mean([int(row["Answer.justification_quality"]) for row in all_samples]))
print("score_fairness:", np.mean([int(row["Answer.score_fairness"]) for row in all_samples]))
print("score_match:", np.mean([int(row["Answer.score_match"]) for row in all_samples]))
