import json
from typing import List
import pandas as pd
from collections import defaultdict


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def create_pandas(data: dict) -> pd.DataFrame:
    qt = []
    guid = []
    question = []
    context = []
    title = []
    ans = []
    p_ans = []
    is_im = []
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                is_impossible = qa.get("is_impossible", False)
                answers = qa["answers"]  # if qa["answers"] else qa["plausible_answers"]
                qt.append(qa.get("question_type", 1))
                guid.append(qa["guid"])
                question.append(qa["question"])
                context.append(paragraph["context"])
                title.append(entry["title"])
                # p_ans.append(qa["plausible_answers"])
                ans.append(answers)
                is_im.append(is_impossible)

    df = pd.DataFrame(
        data={
            "title": title,
            "context": context,
            "guid": guid,
            "is_impossible": is_im,
            "question_type": qt,
            "question": question,
            "answers": ans,
        }
    )
    df["answers"] = df["answers"].apply(concat_answers)
    return df


def concat_answers(answers: List[dict]):
    rtn_dict = defaultdict(list)
    if not answers:
        rtn_dict["answer_start"] = []
        rtn_dict["text"] = []
        return rtn_dict

    for ans in answers:
        for key in ans.keys():
            rtn_dict[key].append(ans[key])
    return rtn_dict
