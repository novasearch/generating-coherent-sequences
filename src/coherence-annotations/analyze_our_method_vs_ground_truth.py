import pandas as pd
import matplotlib.pyplot as plt
import json


keys_file = "keys.json"

with open(keys_file, "r") as f:
    keys = json.load(f)

def _get_rating_a(row):
    for i in range(1, 6):
        if row[f"Answer.ratingA.{i}"]:
            return i
    return None

def _get_rating_b(row):
    for i in range(1, 6):
        if row[f"Answer.ratingB.{i}"]:
            return i
    return None

def load_data(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    df = pd.DataFrame(data)

    # Create the best sequences columns
    df['A'] = df.apply(_get_rating_a, axis=1)
    df['B'] = df.apply(_get_rating_b, axis=1)

    # Select specific columns
    selected_columns = [
        "HITId",
        "WorkerId",
        "Input.recipe_id",
        "Input.title",
        "A",
        "B",
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


def assign_values_our_method(row):
    recipe_id = row['recipe_id']
    ratingA = row['A']
    ratingB = row['B']

    method = keys[str(recipe_id)]["A"] # can be Our Method or Ground-Truth
    if method == "Our Method":
        # Then our method was in A
        return ratingA
    else:
        # If keys "A" has Ground-Truth, our method was in "B"
        return ratingB


def assign_values_ground_truth(row):
    recipe_id = row['recipe_id']
    ratingA = row['A']
    ratingB = row['B']

    method = keys[str(recipe_id)]["A"] # can be Our Method or Ground-Truth
    if method == "Ground-Truth":
        # Then ground-truth was in A
        return ratingA
    else:
        # If keys "A" has Our Method, ground-truth was in "B"
        return ratingB

def plot_histogram(df):
    # Create a histogram
    plt.figure(figsize=(8, 6))
    df['Our Method'].hist(alpha=0.5, label='Our Method')
    df['Ground-Truth'].hist(alpha=0.5, label='Ground-Truth')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Our Method vs Ground-Truth')
    plt.legend()
    plt.savefig("trash/hist.png")


def table(df):
    # Calculate average of 'Our Method' and 'Ground-Truth'
    avg_our_method = round(df['Our Method'].mean(), 2)
    std_our_method = round(df['Our Method'].std(), 2)
    avg_ground_truth = round(df['Ground-Truth'].mean(), 2)
    std_ground_truth = round(df['Ground-Truth'].std(), 2)

    # Create a DataFrame for the averages
    avg_table = pd.DataFrame({
        'Metric': ['Our Method', 'Ground-Truth'],
        'Average': [f"{avg_our_method} \pm {std_our_method}", f"{avg_ground_truth} \pm {std_ground_truth}"]
    })

    print(avg_table)

    return avg_table


if __name__ == '__main__':
    file_path = ".csv"

    df = load_data(file_path)

    df['Our Method'] = df.apply(assign_values_our_method, axis=1)
    df['Ground-Truth'] = df.apply(assign_values_ground_truth, axis=1)

    df = df.drop(["A", "B"], axis=1)
    print(df.head())

    plot_histogram(df)

    #plot_total_picks(df)

    #plot_picks_per_rank(df)

    t = table(df)
    # Print the DataFrame (this can be formatted and used in LaTeX)
    print(t.to_latex(index=False, escape=False))
    # print(t.to_latex(index=False))
    
    #total_count = len(df)
    #d = {
    #    "Error (noSequence) (%)": round(((df[df['noSequence'] == True].shape[0]) / total_count) * 100, 2),
    #    "No error (noSequence) (%)": round(((df[df['noSequence'] == False].shape[0]) / total_count) * 100, 2)
    #}
    #print(d)
