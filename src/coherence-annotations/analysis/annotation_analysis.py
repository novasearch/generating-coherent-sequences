import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


def get_annotation_data(original_data, annotation_columns):
    #columns = ['WorkerId', 'HITId'] + annotation_columns

    # Grab useful columns
    annotation_data = original_data[annotation_columns]

    # TODO when data is updated, use the correct field
    # Extract the method
    annotation_data['Method'] = original_data['Input.image_url'].apply(lambda x: x.split("/")[-2])

    # Create result column with scalar value for annotation
    annotation_data['Result'] = np.where(original_data["Not Coherent"], 0, np.where(original_data["Somewhat Coherent"], 1, np.where(original_data["Very Coherent"], 2, -1)))

    # Create coherent column
    annotation_data["Coherent"] = annotation_data['Result'].apply(lambda x: x > 0)

    return annotation_data


def print_info(data, method, annotation_columns, concise=False):
    print(f"Method «{method}»")
    # Average annotation value
    print(f"Average Score: {data[data['Result'] != -1].mean(numeric_only=True)['Result']:.2f}")

    # percentage_true = (data[annotation_columns+["Coherent"]].mean(numeric_only=True) * 100).round(2)

    if concise:
        percentage_true = (data[["Does Not Apply", "Not Coherent", "Coherent"]].mean(numeric_only=True) * 100).round(2)
    else:
        percentage_true = (data[annotation_columns].mean(numeric_only=True) * 100).round(2)

    print(percentage_true)

    # Create a table
    percentage_table = pd.DataFrame({"Column": percentage_true.index, "Percentage True": percentage_true.values})   

    print(percentage_table)

    plt.clf()
    plt.bar(percentage_true.index, percentage_true.values)
    plt.xlabel("Columns")
    plt.ylabel("Percentage True (%)")
    plt.title(f"Percentage of True Values in Each Column for Method «{method}»")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{method}_{'plot' if not concise else 'plot_concise'}.png")

if __name__ == '__main__':
    data = pd.read_csv(sys.argv[1])
    raters_per_task = int(sys.argv[2])

    raters = len(data["WorkerId"].unique())

    # Number of raters
    print(f"Rated by: {raters}")

    # Count per rater
    print(data["WorkerId"].value_counts())

    gen_errors = data["Answer.generation_error.on"].sum()

    # Number of Images with Problems
    print(f"Generation errors: {gen_errors}")

    data = data[~data["Answer.generation_error.on"]]

    print(f"Acceptable HITs: {data.shape[0]}")

    annotation_rename = {
        "Answer.not_apply.not_apply": "Does Not Apply",
        "Answer.not_coherent.not_coherent": "Not Coherent",
        "Answer.somewhat_coherent.somewhat_coherent": "Somewhat Coherent",
        "Answer.very_coherent.very_coherent": "Very Coherent"
    }

    data.rename(columns=annotation_rename, inplace=True)

    annotation_columns = [
        "Does Not Apply",
        "Not Coherent",
        "Somewhat Coherent",
        "Very Coherent",
    ]

    annotation_data = get_annotation_data(data, annotation_columns)

    method_name = "SD Full Context"

    print_info(annotation_data, method_name, annotation_columns, concise=True)
    print_info(annotation_data, method_name, annotation_columns, concise=False)
