import pandas as pd
import matplotlib.pyplot as plt

KEYS_MAP = {
    "A": "Random Seed",
    "B": "Fixed Seed",
    "C": "Latent 1",
    "D": "Latent 2",
    #"E": "Heuristic",
    "F": "Image-to-Image"
}


def _get_best_sequence(row):
    for col, value in row.items():
        if col.startswith("Answer.bestSequence.") and value:
            return col.split('.')[-1]
    return None  # Return None if no value is True


def _get_second_best_sequence(row):
    for col, value in row.items():
        if col.startswith("Answer.secondBestSequence.") and value:
            return col.split('.')[-1]
    return None  # Return None if no value is True


def _get_third_best_sequence(row):
    for col, value in row.items():
        if col.startswith("Answer.thirdBestSequence.") and value:
            return col.split('.')[-1]
    return None  # Return None if no value is True


def load_data(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    df = pd.DataFrame(data)

    # Create the best sequences columns
    df['bestSequence'] = df.apply(_get_best_sequence, axis=1)
    df['secondBestSequence'] = df.apply(_get_second_best_sequence, axis=1)
    df['thirdBestSequence'] = df.apply(_get_third_best_sequence, axis=1)

    # Select specific columns
    selected_columns = [
        "HITId",
        "WorkerId",
        "Input.recipe_id",
        "Input.title",
        "bestSequence",
        "secondBestSequence",
        "thirdBestSequence",
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


def total_picks_table(df):
    # Extract counts for best, second best, third best, and no sequence
    counts = df[['bestSequence', 'secondBestSequence', 'thirdBestSequence']].apply(pd.Series.value_counts)

    counts = counts.rename(columns=KEYS_MAP)

    counts = counts.drop('E')


def plot_picks_per_rank(df):
    # Extract counts for best, second best, third best, and no sequence
    counts = df[['bestSequence', 'secondBestSequence', 'thirdBestSequence']].apply(pd.Series.value_counts)

    #counts = counts.rename(columns=KEYS_MAP)

    counts = counts.drop('E')

    plt.clf()

    # Create a bar plot
    ax = counts.plot(kind='bar', stacked=False, figsize=(8, 6))
    plt.xlabel('Sequences')
    plt.ylabel('Picks')
    plt.title('Total Number of Picks as Among Best 3')
    plt.xticks(rotation=0)

    sequence_labels = [KEYS_MAP[sequence] for sequence in counts.index]
    ax.set_xticklabels(sequence_labels, rotation=0)

    plt.legend(title='Sequence', loc='upper right')
    plt.tight_layout()

    plt.savefig("plots/vc_picks_per_rank.png")


def count_sequence_picks(data):
    sequences = ['A', 'B', 'C', 'D', 'F']
    sequence_counts = {sequence: 0 for sequence in sequences}

    for col in data.columns:
        if col.startswith('bestSequence') or col.startswith('secondBestSequence') or col.startswith('thirdBestSequence'):
            counts = data[col].value_counts()
            for sequence in sequences:
                if sequence in counts.index:
                    sequence_counts[sequence] += counts[sequence]

    return sequence_counts


def plot_total_picks(df):
    sequence_picks = count_sequence_picks(df)

    plt.clf()

    # Plotting the counts for each sequence
    plt.bar([KEYS_MAP[k] for k in sequence_picks.keys()], sequence_picks.values())
    plt.xlabel('Sequence')
    plt.ylabel('Picks')
    plt.title('Total Picks per Rank')
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("plots/vc_total_picks.png")


def partial_agreement(df):
    partial_agreements = 0

    df = df[df['noSequence']]

    grouped = df.groupby('HITId')
    for _, group in grouped:
        if len(group) <= 1:
            continue
        annotators_sequences = group[['bestSequence', 'secondBestSequence']].values
        for i, annotator1 in enumerate(annotators_sequences):
            for j, annotator2 in enumerate(annotators_sequences):
                if i < j:
                    if ((annotator1 == annotator2).all() or (annotator1[::-1] == annotator2).all()):
                        partial_agreements += 1
                        break

    return partial_agreements


def full_agreement(df):
    full_agreements = 0
    
    grouped = df.groupby('HITId')
    for hit_id, group in grouped:
        annotators_sequences = group[['bestSequence', 'secondBestSequence']].values
        unique_sequences = set(map(tuple, annotators_sequences))
        if len(unique_sequences) == 1:
            full_agreements += 1
    
    return full_agreements


def table_summarize(df):
    counts = df[['bestSequence', 'secondBestSequence', 'thirdBestSequence']].apply(pd.Series.value_counts)
    counts = counts.drop('E')
    counts = counts.fillna(0)

    percentages = counts.apply(lambda x: round((x / x.sum()) * 100, 2))

    print(counts)

    sequence_picks = count_sequence_picks(df)
    print(sequence_picks)

    data = {
        "Sequence": KEYS_MAP.values(),
        "Total Picks": sequence_picks.values(),
        "Best (%)": percentages['bestSequence'],
        "Second Best (%)": percentages['secondBestSequence'],
        "Third Best (%)": percentages['thirdBestSequence'],
    }

    return pd.DataFrame(data)


if __name__ == '__main__':
    file_path = ".csv"

    df = load_data(file_path)

    #print(partial_agreement(df))
    #print(full_agreement(df))

    #plot_total_picks(df)

    #plot_picks_per_rank(df)

    t = table_summarize(df)
    print(t.to_latex(index=False))
    
    total_count = len(df)
    d = {
        "Error (noSequence) (%)": round(((df[df['noSequence'] == True].shape[0]) / total_count) * 100, 2),
        "No error (noSequence) (%)": round(((df[df['noSequence'] == False].shape[0]) / total_count) * 100, 2)
    }
    print(d)
