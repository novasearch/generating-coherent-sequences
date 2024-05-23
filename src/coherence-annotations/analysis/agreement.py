import pandas as pd
import numpy as np
import sys
import fleiss_kappa as fleiss

data = pd.read_csv(sys.argv[1])
raters_per_task = int(sys.argv[2])

data = data[~data["Answer.generation_error.on"]]

annotation_columns = [
    "Answer.not_coherent.not_coherent",
    "Answer.somewhat_coherent.somewhat_coherent",
    "Answer.very_coherent.very_coherent",
    "Answer.not_apply.not_apply"
]


def get_annotation_data(original_data):

    # Creating a result column
    columns = ['WorkerId', 'HITId'] + annotation_columns

    # Grab useful columns
    annotation_data = original_data[columns]

    # Create result column with scalar value for annotation
    annotation_data['Result'] = np.where(original_data[annotation_columns[0]], 0, np.where(original_data[annotation_columns[1]], 1, np.where(original_data[annotation_columns[2]], 2, -1)))

    return annotation_data

annotation_data = get_annotation_data(data)

count = annotation_data.groupby(["HITId", 'Result'])['WorkerId'].nunique().reset_index()

all_agree = count.loc[count["WorkerId"] == raters_per_task]

all_agree_percent = all_agree.shape[0] * 100 / len(annotation_data["HITId"].unique())

print(f"Full agreement: {all_agree_percent / 100:.2%}")

majority = count.loc[count["WorkerId"] >= int(raters_per_task/2)+1]
majority_percent = majority.shape[0] * 100 / len(annotation_data["HITId"].unique())

print(f"Majority agree: {majority_percent/100:.2%}")

# Count the number of True values for each column, grouped by HITId
counts_df = annotation_data.groupby('HITId')[annotation_columns].sum().reset_index()

# Display the counts_df as a matrix
matrix_form = counts_df[annotation_columns].values

# Filter out HITs without the minimum number of raters
matrix_form = matrix_form[matrix_form.sum(axis=1) == raters_per_task]

counts_collapsed = counts_df

counts_collapsed["Coherent"] = counts_collapsed["Answer.somewhat_coherent.somewhat_coherent"] + counts_collapsed["Answer.very_coherent.very_coherent"]

# Display the counts_df as a matrix
matrix_form_collapsed = counts_collapsed[["Answer.not_coherent.not_coherent", "Coherent"]].values

matrix_form_collapsed = matrix_form_collapsed[matrix_form_collapsed.sum(axis=1) == raters_per_task]

print("General Agreement:")
general_agreement = fleiss.fleissKappa(matrix_form.tolist(), raters_per_task)

print("Collapsed Labels Agreement:")
collapsed_agreement = fleiss.fleissKappa(matrix_form_collapsed.tolist(), raters_per_task)