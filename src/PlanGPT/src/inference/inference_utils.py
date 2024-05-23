import json
import os

import pandas as pd
from bert_score import score
from jury import Jury


def compute_metrics(outputs, references, metrics=["bleu", "meteor", "rouge"]):
    jury_metrics = metrics.copy()

    if "accuracy" in metrics:
        jury_metrics.remove("accuracy")
    if "bertscore" in metrics:
        jury_metrics.remove("bertscore")

    scorer = Jury(metrics=metrics)

    outputs = ["gfgfgfgfg" if o == "" else o for o in outputs]
    scores = scorer(predictions=[[o] for o in outputs], references=[[s] for s in references])

    if "accuracy" in metrics:
        acc = [1.0 if r.strip().lower() in o.strip().lower() else 0.0 for o, r in zip(outputs, references)]
        acc_mean = sum(acc) / len(acc)
        scores["accuracy"] = acc_mean

    if "bertscore" in metrics:
        P, R, F1 = score(outputs, references, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=True)

        mean_p = sum(P.tolist()) / len(P.tolist())
        mean_r = sum(R.tolist()) / len(R.tolist())
        mean_f1 = sum(F1.tolist()) / len(F1.tolist())

        scores['bertscore'] = {"recall": mean_r, "precision": mean_p, "f1": mean_f1, "model_type": "microsoft/deberta-xlarge-mnli"}

    return scores


def update_log_file(ckpt_path):
    base_folder = os.path.dirname(ckpt_path)
    if "checkpoint" not in ckpt_path:
        base_folder = ckpt_path  # when the ckpt is the final version
    log_file = os.path.join(base_folder, "metrics_log.csv")
    ckpt_folders = [f for f in next(os.walk(base_folder))[1] if "checkpoint" in f]

    data = []

    for ckpt in ckpt_folders:
        ckpt_metrics = [os.path.join(base_folder, ckpt, f) for f in next(os.walk(os.path.join(base_folder, ckpt)))[2] if f.endswith("metrics.json")]
        if len(ckpt_metrics) == 0:
            print(f"Skipping {ckpt} because no metrics file was found")
            continue
        with open(ckpt_metrics[0], "r") as infile:
            ckpt_scores = json.load(infile)
        ckpt_data = {"run_name": ckpt, "bleu": ckpt_scores["bleu"]["score"], "meteor": ckpt_scores["meteor"]["score"],
                     "rouge": ckpt_scores["rouge"]["rougeL"], "accuracy": ckpt_scores["accuracy"]}
        data.append(ckpt_data)

    print(f"Writing to {log_file}")

    df = pd.DataFrame(data)

    df.to_csv(log_file)




