import pandas as pd
import matplotlib.pyplot as plt

KEYS_MAP = {
    "A": "Latent 1 > Heuristic",
    "B": "Heuristic > Latent 1",
    "C": "Tie"
}


def _get_best_sequence(row):
    for col, value in row.items():
        if col.startswith("Answer.bestSequence.") and value:
            return col.split('.')[-1]
    return None  # Return None if no value is True


def load_data(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    df = pd.DataFrame(data)

    # Create the best sequences columns
    df['bestSequence'] = df.apply(_get_best_sequence, axis=1)

    # Select specific columns
    selected_columns = [
        "HITId",
        "WorkerId",
        "Input.recipe_id",
        "Input.title",
        "bestSequence",
        "Answer.noGoodSequence.on"
    ]

    # Extract the selected columns into a new DataFrame
    df = df[selected_columns]

    # Renaming columns: removing "Input" and mapping Answer.bestSequence.* columns
    df = df.rename(columns={
        "Input.recipe_id": "recipe_id",
        "Input.title": "title",
        "Answer.noGoodSequence.on": "noSequence"
    })

    return df


def plot_total_picks(df):
    counts = df[['bestSequence']].apply(pd.Series.value_counts)
    total_picks = counts.sum()
    percentages = counts / total_picks * 100  # Calculate percentages

    percentages = percentages.rename(index=KEYS_MAP)

    print(percentages)

    plt.clf()

    # Create a bar plot
    ax = percentages.plot(kind='bar', stacked=False, figsize=(8, 6))
    plt.xlabel('Sequences')
    plt.ylabel('Percentage of Picks')
    plt.title('Win Percentage')
    plt.xticks(rotation=0)

    #sequence_labels = [KEYS_MAP[sequence] for sequence in percentages.index]
    #ax.set_xticklabels(sequence_labels, rotation=0)

    plt.tight_layout()

    #plt.savefig("plots/heuristic_vs_latent_total_picks_percentage.png")

def table(df):

    total_examples = len(df)

    # Extract noSequence occurrences
    noSequence_count = len(df[df['noSequence']])

    df = df[df['noSequence'] == False]

    counts = df[['bestSequence']].apply(pd.Series.value_counts)
    print(counts)
    total_picks = counts.sum()
    print(total_examples, noSequence_count, total_picks)
    percentages = round(counts / total_examples * 100, 2)  # Calculate percentages

    percentages = percentages.rename(index=KEYS_MAP, columns={"bestSequence": 'Win %'})

    percentages.loc["No Good Sequence"] = round(noSequence_count / total_examples * 100, 2)

    return percentages


if __name__ == '__main__':
    file_path = ".csv"

    df = load_data(file_path)

    #plot_total_picks(df)
    t = table(df)
    print(t.to_latex())