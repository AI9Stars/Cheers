# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

df = pd.read_json(args.filename, orient="records", lines=True)

print("Summary")
print("=======")
print(f"Total images: {len(df)}")
print(f"Total prompts: {len(df.groupby('metadata'))}")
print(f"% correct images: {df['correct'].mean():.2%}")
print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
print()

task_scores = []
result = []
result.append( {"key": args.filename})
print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
    json_item = {
            "tag": tag,
            "score": f"{task_df['correct'].mean()}",
            "total_correct": f"{task_df['correct'].sum()}",
            "total_tasks": f"{len(task_df)}"
        }
    result.append(json_item)
print()

print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")
