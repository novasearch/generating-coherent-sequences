import argparse
import json

import pandas as pd
from bert_score import score
from jury import Jury
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("Train an Transformer model")
    # General params
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--intent", action="store_true")
    parser.add_argument("--build_csv", action="store_true")

    return parser.parse_args()


def score_predictions(predictions, targets):
    print("Computing metrics....")

    acc = [1.0 if t.strip().lower() in o.strip().lower() else 0.0 for o, t in zip(predictions, targets)]
    acc_mean = sum(acc) / len(acc)

    scorer = Jury(metrics=["bleu", "meteor", "rouge"])
    # print([[o] for o in outputs])
    predictions = ["### Response:" if o == "" else o for o in predictions]

    scores = scorer(predictions=[[o] for o in predictions], references=[[s] for s in targets])
    scores["accuracy"] = acc_mean

    P, R, F1 = score(predictions, targets, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=True)

    mean_p = sum(P.tolist()) / len(P.tolist())
    mean_r = sum(R.tolist()) / len(R.tolist())
    mean_f1 = sum(F1.tolist()) / len(F1.tolist())

    scores["bertscore_custom"] = {"recall": mean_r, "precision": mean_p, "f1": mean_f1,
                                  "model_type": "microsoft/deberta-xlarge-mnli"}

    return scores


def main():
    args = get_args()

    # predictions_file = "checkpoint_2500_outputs.json"
    predictions_file = args.predictions_file

    if args.build_csv:

        with open(predictions_file, "r") as infile:
            results_dicts = json.load(infile)

        navigational_intents = ['AMAZON.PreviousIntent', 'AMAZON.RepeatIntent', 'AMAZON.StopIntent', 'AMAZON.YesIntent',
                                'NextStepIntent', 'null', 'PreviousStepIntent', 'ResumeTaskIntent']
        other_intents = ['AMAZON.FallbackIntent', 'CommonChitChatIntent', 'IdentifyProcessIntent', 'MoreDetailIntent']
        nav_item_count = 0
        other_item_count = 0
        navigational_data = []
        other_data = []
        curiosity_data = {
            "intent": 'GetCuriositiesIntent',
            "item_count": results_dicts['GetCuriositiesIntent']["total_items"],
            "bleu": results_dicts['GetCuriositiesIntent']["bleu"]['score'],
            "meteor": results_dicts['GetCuriositiesIntent']["meteor"]["score"],
            "rouge": results_dicts['GetCuriositiesIntent']["rouge"]['rougeL'],
            "bertscore": results_dicts['GetCuriositiesIntent']["bertscore_custom"]["f1"],
            "accuracy": results_dicts['GetCuriositiesIntent']["accuracy"],
        }
        ing_replacement_data = {
            "intent": 'IngredientsReplacementIntent',
            "item_count": results_dicts['IngredientsReplacementIntent']["total_items"],
            "bleu": results_dicts['IngredientsReplacementIntent']["bleu"]['score'],
            "meteor": results_dicts['IngredientsReplacementIntent']["meteor"]["score"],
            "rouge": results_dicts['IngredientsReplacementIntent']["rouge"]['rougeL'],
            "bertscore": results_dicts['IngredientsReplacementIntent']["bertscore_custom"]["f1"],
            "accuracy": results_dicts['IngredientsReplacementIntent']["accuracy"],
        }
        definition_data = {
            "intent": 'ARTIFICIAL.DefinitionQuestionIntent',
            "item_count": results_dicts['ARTIFICIAL.DefinitionQuestionIntent']["total_items"],
            "bleu": results_dicts['ARTIFICIAL.DefinitionQuestionIntent']["bleu"]['score'],
            "meteor": results_dicts['ARTIFICIAL.DefinitionQuestionIntent']["meteor"]["score"],
            "rouge": results_dicts['ARTIFICIAL.DefinitionQuestionIntent']["rouge"]['rougeL'],
            "bertscore": results_dicts['ARTIFICIAL.DefinitionQuestionIntent']["bertscore_custom"]["f1"],
            "accuracy": results_dicts['ARTIFICIAL.DefinitionQuestionIntent']["accuracy"],
        }
        question_data = {
            "intent": 'QuestionIntent',
            "item_count": results_dicts['QuestionIntent']["total_items"],
            "bleu": results_dicts['QuestionIntent']["bleu"]['score'],
            "meteor": results_dicts['QuestionIntent']["meteor"]["score"],
            "rouge": results_dicts['QuestionIntent']["rouge"]['rougeL'],
            "bertscore": results_dicts['QuestionIntent']["bertscore_custom"]["f1"],
            "accuracy": results_dicts['QuestionIntent']["accuracy"],
        }
        sensitive_data = {
            "intent": 'ARTIFICIAL.SensitiveIntent',
            "item_count": results_dicts['ARTIFICIAL.SensitiveIntent']["total_items"],
            "bleu": results_dicts['ARTIFICIAL.SensitiveIntent']["bleu"]['score'],
            "meteor": results_dicts['ARTIFICIAL.SensitiveIntent']["meteor"]["score"],
            "rouge": results_dicts['ARTIFICIAL.SensitiveIntent']["rouge"]['rougeL'],
            "bertscore": results_dicts['ARTIFICIAL.SensitiveIntent']["bertscore_custom"]["f1"],
            "accuracy": results_dicts['ARTIFICIAL.SensitiveIntent']["accuracy"],
        }

        for intent in results_dicts:
            res = results_dicts[intent]
            metrics = {
                "intent": intent,
                "item_count": res["total_items"],
                "bleu": res["bleu"]['score'],
                "meteor": res["meteor"]["score"],
                "rouge": res["rouge"]['rougeL'],
                "bertscore": res["bertscore_custom"]["f1"],
                "accuracy": res["accuracy"],
            }
            if intent in navigational_intents:
                nav_item_count += res["total_items"]
                navigational_data.append(metrics)
            elif intent in other_intents:
                other_item_count += res["total_items"]
                other_data.append(metrics)

        final_nav_data = {"intent": 'navigational', "item_count": nav_item_count, "bleu": 0.0, "meteor": 0.0,
                          "rouge": 0.0, "bertscore": 0.0, 'accuracy': 0.0}
        for intent_metrics in navigational_data:
            ratio = intent_metrics["item_count"] / nav_item_count
            final_nav_data['bleu'] += intent_metrics['bleu'] * ratio
            final_nav_data['meteor'] += intent_metrics['meteor'] * ratio
            final_nav_data['rouge'] += intent_metrics['rouge'] * ratio
            final_nav_data['bertscore'] += intent_metrics['bertscore'] * ratio
            final_nav_data['accuracy'] += intent_metrics['accuracy'] * ratio

        final_other_data = {"intent": 'other', "item_count": other_item_count, "bleu": 0.0, "meteor": 0.0,
                            "rouge": 0.0, "bertscore": 0.0, 'accuracy': 0.0}
        for intent_metrics in other_data:
            ratio = intent_metrics["item_count"] / other_item_count
            final_other_data['bleu'] += intent_metrics['bleu'] * ratio
            final_other_data['meteor'] += intent_metrics['meteor'] * ratio
            final_other_data['rouge'] += intent_metrics['rouge'] * ratio
            final_other_data['bertscore'] += intent_metrics['bertscore'] * ratio
            final_other_data['accuracy'] += intent_metrics['accuracy'] * ratio

        data = [curiosity_data, ing_replacement_data, definition_data, question_data, final_nav_data, final_other_data,
                sensitive_data]

        total_item_count = sum([d["item_count"] for d in data])
        for d in data:
            d["item_percentage"] = d["item_count"] / total_item_count

        df = pd.DataFrame(data)
        df.to_csv(args.predictions_file.replace("_intent_metrics.json", "_grouped_intent_metrics.csv"), index=False)

        exit(0)

    output_file = predictions_file.replace("_outputs.json", "").replace("predictions.json", "")

    with open(predictions_file, "r") as infile:
        prediction_dicts = json.load(infile)

    if args.intent:
        print(output_file)
        intent_scores = dict()
        intents = list(set([p["current_intent"] for p in prediction_dicts]))

        for intent in tqdm(intents):
            predictions = [p["prediction"] for p in prediction_dicts if p["current_intent"] == intent]
            targets = [p["target"] for p in prediction_dicts if p["current_intent"] == intent]

            assert len(predictions) == len(targets)

            scores = score_predictions(predictions, targets)

            intent_scores[intent] = scores

        file_path = output_file + "_intent_metrics.json"
        del scores
        scores = intent_scores

    else:
        predictions = [p["prediction"] for p in prediction_dicts]
        targets = [p["target"] for p in prediction_dicts]

        scores = score_predictions(predictions, targets)
        file_path = output_file + "_metrics.json"

    with open(file_path, "w+") as score_file:
        json.dump(scores, score_file, indent=4)
    print(scores)


if __name__ == "__main__":
    main()
