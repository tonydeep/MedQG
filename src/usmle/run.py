"""
MedQG Question Generation - LangGraph Refactored Version

This module provides the main entry points for USMLE question generation
using LangGraph-based orchestration.
"""

import json
import os
from tqdm import tqdm
from src.utils import retry_parse_fail_prone_cmd
from src.usmle.question_graph import QuestionGenerationGraph

ENGINE = os.getenv("ENGINE")


@retry_parse_fail_prone_cmd
def autofb_usmleqgen(clinical_note: str, keypoint: str, topic: str, max_attempts: int) -> list:
    """
    Generate USMLE question with automatic feedback and refinement using LangGraph.

    Args:
        clinical_note: Clinical note text
        keypoint: Test keypoint/concept
        topic: USMLE topic
        max_attempts: Maximum refinement iterations

    Returns:
        List of iteration results with feedback
    """
    graph = QuestionGenerationGraph(engine=ENGINE)
    return graph.generate(clinical_note, keypoint, topic, max_attempts)
def run_cmd():
    concepts = sys.argv[2:]
    max_attempts = 5
    content_to_fb = autofb_usmleqgen(
        concepts=concepts,
        max_attempts=max_attempts,
    )

    res = []
    for s in  content_to_fb:
        sent = s["sentence"]
        fb = ";  ".join(s["concept_feedback"]) + " " + s["commonsense_feedback"]
        res.append(f"{sent} ({fb})")
    print(" -> ".join(res))


def run_iter(inputs_file_path: str, max_attempts: int = 4):
    test_df = pd.read_json(inputs_file_path, orient="records")
    is_rerun = "status" in test_df.columns
    if not is_rerun:
        test_df["content_to_fb"] = None
        test_df["content_to_fb"] = test_df["content_to_fb"].astype(object)
        test_df["status"] = None
        #this is a test comment

    else:
        print("Status column already exists! Looks like you're trying to do a re-run")
        print(test_df["status"].value_counts())
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running autofb iter"):
        if row["status"] == "success":
            continue
        content_to_fb = autofb_usmleqgen(clinical_note=row["clinical_note"],keypoint=row['keypoint'],topic=row['topic'], max_attempts=max_attempts)
            
        dict_to_write = {"clinical_note":row["clinical_note"],"keypoint":row['keypoint'],"topic":row['topic'],"content_to_fb":content_to_fb}
        output_path = inputs_file_path + (".iter.out" if not is_rerun else ".v0")
        version = 1
        output_path = output_path + f".v{version}"
        with open(output_path, 'a+') as f:
            json.dump(dict_to_write,f)
            f.write('\n')
        print(f"content to fb : {content_to_fb}")
        test_df.at[i, "content_to_fb"] = content_to_fb
        test_df.at[i, "status"] = "success"


if __name__ == "__main__":
    import sys
    import pandas as pd

    if sys.argv[1] == "cmd":
        run_cmd()

    elif sys.argv[1] == "batch-iter":
        run_iter(inputs_file_path=sys.argv[2])

    else:
        raise ValueError("Invalid mode: choose between cmd, batch-iter, batch-multi")


