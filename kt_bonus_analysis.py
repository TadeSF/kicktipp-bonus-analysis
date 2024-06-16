"""
This CLI tool is used to analyze the bonus data for a given year for Kicktipp users.

Usage:
    bonus_analysis.py FILE [--exclude=<players>]
    bonus_analysis.py FILE MODE [COLUMN] [--exclude=<players>]


Arguments:
    FILE                    The file containing the bonus data for the Kicktipp users.
    MODE                    The mode of analysis to perform on the data. (`prediction_freq`, `groups`, `prediction_network`, `similarity_network`)
    COLUMN                  The column to analyze in the data. Must match the column name in the file. (applicable only in `prediction_freq` mode)

Options:
    -h --help               Show this screen.
    --version               Show version.
    --exclude=<players>     List of players to exclude from the analysis. (separated by comma without spaces)
"""

import os
import re
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from docopt import docopt

def check_columns(data):
    COLUMN_GROUP_REGEX = r'^Gr [ABCDEFGHIJKLMNO]$'
    for col in data.columns:
        if col in ['Name', 'TipperID']:
            continue
        if col in ['WM', 'EM', 'Tor', 'HF']:
            continue
        if not re.match(COLUMN_GROUP_REGEX, col):
            raise ValueError(f"Invalid column found: {col}.")
    return True

def clean_data(data, excluded_players=[]):
    data = data.dropna()

    # Convert HF datatype from string to array (delimiter: ' ' )
    data['HF'] = sorted(data['HF'].str.split(' '))
    
    for player in excluded_players:
        data = data[data['Name'] != player]
    
    return data

def plot_prediction_frequencies(data, column_name):
    if column_name == 'HF':
        prediction_counts = unpack_semifinalists(data[column_name])
    else:
        prediction_counts = data[column_name].value_counts()

    # Creating the subplot with 1 row and 2 columns
    _fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Bar chart of the predictions
    prediction_counts.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Frequency of Predictions for {column_name}')
    axes[0].set_xlabel('Teams')
    axes[0].set_ylabel('Counts')

    # Pie chart of the predictions
    prediction_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
    axes[1].set_ylabel('')  # Clear the y-axis label on pie chart
    axes[1].set_title(f'Percentage Distribution for {column_name}')

    plt.tight_layout()
    plt.show()
    
def unpack_semifinalists(semifinalists):
    semifinalist_counts = Counter(team for sublist in semifinalists for team in sublist)
    return pd.Series(semifinalist_counts).sort_values(ascending=False)

def plot_group_winner_predictions(data):
    group_columns = [col for col in data.columns if col.startswith('Gr')]
    _fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    
    for idx, column in enumerate(group_columns):
        ax = axes[idx//3, idx%3]
        group_counts = data[column].value_counts()
        group_counts.plot(kind='bar', ax=ax, color='lightblue')
        ax.set_title(f'Predictions for Group {column[-1]} Winner')
        ax.set_xlabel('Teams')
        ax.set_ylabel('Counts')
    
    plt.tight_layout()
    plt.show()

def plot_prediction_network(data):
    G = nx.Graph()

    # Prepare a Counter to tally occurrences of each team in the semi-finals
    team_counter = Counter(team for teams in data['HF'] for team in teams)

    # Adding nodes with dynamic size based on predictions count
    for team, count in team_counter.items():
        G.add_node(team, size=count * 100)  # Scale factor can be adjusted

    # Prepare a defaultdict to tally edges weight (connection strength)
    edge_counter = defaultdict(int)
    for teams in data['HF']:
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                if teams[i] != teams[j]:
                    edge_pair = tuple(sorted([teams[i], teams[j]]))  # Sort to avoid duplicate edges like (A, B) and (B, A)
                    edge_counter[edge_pair] += 1

    # Adding edges with dynamic width based on how often pairs are predicted together
    for (team1, team2), count in edge_counter.items():
        G.add_edge(team1, team2, weight=count)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    # Draw nodes with sizes adjusted according to their prediction count
    node_sizes = [G.nodes[node]['size'] for node in G]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)

    # Draw edges with dynamic widths
    edge_widths = [0.3 + G[u][v]['weight']*0.3 for u, v in G.edges()]  # Adjust the width scaling factor as needed
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='black')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_family="sans-serif")

    plt.title('Network of Teams Predicted to Reach the Semi-finals Together')
    plt.axis('off')  # Turn off the axis
    plt.show()
    
def normalize_dataframe(df):
    # Extracting 'Name' and 'TipperID'
    identifiers = df[['Name', 'TipperID']]
    
    # One-hot encode categorical columns except 'HF'
    categorical_cols = ['Tor', 'Gr A', 'Gr B', 'Gr C', 'Gr D', 'Gr E', 'Gr F', 'EM']
    df_encoded = pd.get_dummies(df[categorical_cols])

    # Handle the 'HF' column by creating a flat list of all possible teams
    all_teams = set(team for sublist in df['HF'] for team in sublist)
    for team in all_teams:
        df_encoded[f'HF_{team}'] = df['HF'].apply(lambda x: 1 if team in x else 0)

    # Combine identifiers with the encoded data
    result_df = pd.concat([identifiers, df_encoded], axis=1)

    return result_df
    
def create_player_similarity_network(data):    
    # Calculate cosine similarity matrix from the DataFrame
    similarity_matrix = cosine_similarity(data.drop(['Name', 'TipperID'], axis=1))
    
    # min-max normalize the similarity matrix to a range of [0, 1]
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
    
    # Create a graph
    G = nx.Graph()

    # Add edges between all players weighted by adjusted similarity score
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            similarity_weight = similarity_matrix[i][j]
            if similarity_weight > 0:  # Optional: filter out very low similarities if necessary
                # Add an edge with a weight that inversely scales with similarity for layout purposes
                G.add_edge(data.iloc[i]['Name'], data.iloc[j]['Name'], weight=(similarity_weight))

    # Use the spring layout with customized distance (weight inversely related to similarity)
    pos = nx.spring_layout(G, weight='weight', iterations=100, scale=1) # More iterations for a stable layout

    plt.figure(figsize=(20, 20))
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    # Draw nodes with the node size scaled by a factor (optional)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)

    # Draw edges with widths that decrease with increasing dissimilarity
    edge_widths = [3 * (w - 0.3) if w > 0.4 else 0 for w in weights]  # Using logarithmic scale to normalize the visualization
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, alpha=1, edge_color="lightblue", label=weights)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')

    plt.title('Network of Player Prediction Similarity')
    plt.axis('off')  # Turn off the axis to enhance the focus on the graph
    plt.show()



def main(arguments):
    
    data = None

    if arguments['FILE']:
        # if path is relative, convert to absolute path
        file_path = arguments['FILE']
        if not file_path.startswith('/'):
            file_path = os.path.abspath(file_path)
    
        print("Reading data from file:")
        print(file_path)
    
        data = pd.read_csv(file_path, sep=';')
        if not check_columns(data):
            raise ValueError("Invalid columns found in the data.")
        data = clean_data(data, arguments['--exclude'].split(',') if '--exclude' in arguments else [])
        
        print("Data read successfully.")
        print("\nData columns:\n")
        print(data)
        
        for col in filter(lambda x: x not in ['Name', 'TipperID'], data.columns):
            print(f"\nUnique values for '{col}':")
            if col == 'HF':
                print(unpack_semifinalists(data[col]))
                continue
            print(data[col].value_counts())
            
    if arguments['FILE'] and arguments['MODE'] == 'prediction_freq' and arguments['COLUMN']:
        if arguments['COLUMN'] not in data.columns:
            raise ValueError(f"Column {arguments['COLUMN']} not found in the data.")
        
        plot_prediction_frequencies(data, arguments['COLUMN'])
        
    if arguments['FILE'] and arguments['MODE'] == 'groups' and not arguments['COLUMN']:
        plot_group_winner_predictions(data)
        
    if arguments['FILE'] and arguments['MODE'] == 'prediction_network' and not arguments['COLUMN']:
        plot_prediction_network(data)
        
    if arguments['FILE'] and arguments['MODE'] == 'similarity_network' and not arguments['COLUMN']:
        df_encoded = normalize_dataframe(data)
        create_player_similarity_network(df_encoded)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Bonus Analysis 1.0')
    main(arguments)