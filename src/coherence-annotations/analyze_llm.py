import pandas as pd
import matplotlib.pyplot as plt

#NOTE_DICT = {
#    "A": "Copy Problem",
#    "B": "Step has too many actions",
#    "C": "Good Prompt despite too many actions",
#    "D": "Prompt has Hallucinations"
#}

NOTE_DICT = {
    "A": "Prompt copied\nfrom input",
    "B": "Step with too many actions",
    "C": "Good Prompt despite B",
    "D": "Hallucinations"
}


def load_and_select_columns(file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    data = data.rename(columns={data.columns[3]: 'Rating'})
    data.dropna(subset=['Rating'], inplace=True, how='any')
    data['Rating'] = data['Rating'].astype(int)
    data = data.rename(columns={data.columns[4]: 'Note'})


    # Select the columns 'Id', 'Rating', and 'Note'
    selected_columns = data[["Id", "Rating", "Note"]]
    
    return selected_columns


def calculate_average_rating(data, clean_errors=False):
    df = data.copy()

    if clean_errors:
        # Filter out rows where 'Rating' equals -1 (errors)
        df = df[df['Rating'] != -1]

    # Calculate the average rating
    average_rating = df['Rating'].mean()
    return average_rating


def plot_averages(data_dict, clean_errors=False):
    # Plotting the averages for different files in the same plot
    plt.figure(figsize=(10, 6))
    for file_name, data in data_dict.items():
        average = calculate_average_rating(data, clean_errors)
        plt.bar(file_name, average, label=file_name)

    plt.xlabel('Files')
    plt.ylabel('Average Rating')
    plt.title('Average Rating Comparison')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f".png")
    plt.show()

def plot_rating_bar(data, title):
    # Plot a bar plot of the 'Rating' column
    plt.figure(figsize=(8, 6))
    data['Rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(f'Rating Distribution for {title}')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.savefig(f".png")
    plt.show()


def plot_errors_bar(data, title):

    data_copy = data.copy()
    # Map the 'Note' column values using NOTE_DICT
    data_copy['Note'] = data_copy['Note'].map(NOTE_DICT)

    # Plot a bar plot of the updated 'Note' column
    plt.figure(figsize=(8, 6))
    data_copy['Note'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(f'Note Distribution for {title}')
    plt.xlabel('Note')
    plt.xticks(rotation=10)
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.savefig(f".png")
    plt.show()

def calculate_average_rating_by_error(data):
    data_copy = data.copy()

    # Map the 'Note' column values using NOTE_DICT
    data_copy['Note'] = data_copy['Note'].map(NOTE_DICT)

    # Calculate average ratings for each error type
    avg_ratings = data_copy.groupby('Note')['Rating'].mean().reset_index()
    
    return avg_ratings

def plot_average_ratings_by_error(avg_ratings, title):
    # Plotting the average ratings for each error type
    plt.figure(figsize=(8, 6))
    plt.bar(avg_ratings['Note'], avg_ratings['Rating'], color='skyblue')
    plt.xlabel('Error Types')
    plt.xticks(rotation=10)
    plt.ylabel('Average Rating')
    plt.title(f'Average Ratings by Error Types for {title}')
    plt.ylim(0, 4)  # Set y-axis limit from 0 to 4 (assuming ratings are from 0 to 4)
    plt.grid(axis='y')
    plt.savefig(f".png")
    plt.show()

def calculate_statistics(data):
    # Calculate average rating
    avg_rating = round(data['Rating'].mean(), 2)

    # Calculate average rating removing errors (Rating = -1)
    avg_rating_no_errors = round(data[data['Rating'] != -1]['Rating'].mean(), 2)

    # Calculate standard deviation of rating
    rating_std_dev = round(data['Rating'].std(), 2)

    # Calculate standard deviation of rating removing errors (Rating = -1)
    rating_std_dev_no_errors = round(data[data['Rating'] != -1]['Rating'].std(), 2)
    
    return avg_rating, avg_rating_no_errors, rating_std_dev, rating_std_dev_no_errors


def errors_by_rating(df):
    notes = NOTE_DICT.keys()

    # Initialize an empty dictionary to store counts
    counts = {note: [] for note in notes}

    # Loop through each rating and note combination to count occurrences
    for rating in sorted(df['Rating'].unique()):
        rating_counts = df[df['Rating'] == rating]['Note'].value_counts().to_dict()
        for note in notes:
            counts[note].append(rating_counts.get(note, 0))

    # Create a new DataFrame with counts
    counts_df = pd.DataFrame(counts, index=sorted(df['Rating'].unique()))

    # Rename index and columns for better readability
    counts_df.index.name = 'Rating'
    counts_df.columns = [NOTE_DICT[note] for note in notes]

    return counts_df


import os

if __name__ == '__main__':
    data_dict = {}  # Dictionary to store dataframes

    base_dir = ""
    file_paths = ["c0c.csv", "c0p.csv", "c1c.csv", "c1p.csv", "c2c.csv", "c2p.csv"]

    for file_path in file_paths:
        full_path = os.path.join(base_dir, file_path)
        file_name = file_path.replace('.csv', '')  # Extract file name without extension
        selected_data = load_and_select_columns(full_path)
        data_dict[file_name] = selected_data

    results = []

    # Plots
    #for file_name, data_copy in data_dict.items():
    #    plot_rating_bar(data_copy, file_name)
    #    plot_errors_bar(data_copy, file_name)
    #    avg_ratings = calculate_average_rating_by_error(data_copy)
    #    plot_average_ratings_by_error(avg_ratings, file_name)

    # Table
    #for file_name, data_copy in data_dict.items():
    #    avg_rating, avg_rating_no_errors, rating_std_dev, rating_std_dev_no_errors = calculate_statistics(data_copy)
    #
    #    results.append({
    #        'File': file_name,
    #        'Average Rating': f"{avg_rating:.2f} \pm {rating_std_dev:.2f}",
    #        'Average Rating (excluding errors)': f"{avg_rating_no_errors:.2f} \pm {rating_std_dev_no_errors:.2f}"
    #    })

    # Errors
    for file_name, data_copy in data_dict.items():

        data_copy['Note'] = data_copy['Note'].where(data_copy['Note'].isin(NOTE_DICT.keys()), None)

        # Create a pivot table
        print(file_name)

        total_count = len(data_copy)
        d = {
            "Error (%)": round(((data_copy['Rating'] == -1).sum() / total_count) * 100, 2),
            "No error (%)": round(((data_copy['Rating'] != -1).sum() / total_count) * 100, 2)
        }
        print(d)

        pivot = pd.pivot_table(data_copy, 
                        index='Rating', 
                        columns='Note', 
                        aggfunc='size', 
                        fill_value=0)

        #print(pivot.to_latex())

    # Plotting the averages for different files in the same plot
    #plot_averages(data_dict)
    #plot_averages(data_dict, True)
    #results_df = pd.DataFrame(results)
    #latex_table = results_df.to_latex(index=False, escape=False, column_format='|c|c|')
    #print(results_df.to_latex())
    #print(latex_table)