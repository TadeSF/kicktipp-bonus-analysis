# Bonus Analysis Tool for Kicktipp Users

This Python script provides a command-line interface (CLI) tool to analyze bonus data of Kicktipp users. It supports various modes of analysis such as prediction frequencies, group winner predictions, and prediction networks. Users can exclude specific players from the analysis if desired.

## Features

- Analyze the frequency of predictions in specific columns.
- View group winner predictions across different groups.
- Create a network graph to illustrate the relationships between teams predicted to reach the semi-finals together.

## Requirements

- Python 3.6+
- Pandas
- Matplotlib
- NetworkX
- Docopt

## Installation

To use this tool, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TadeSF/kicktipp-bonus-analysis.git
   ```
2. Navigate to the directory where the repository is cloned:
   ```bash
   cd bonus_analysis
   ```
3. Ensure you have Python installed, and install the required packages:
   ```bash
   pip install pandas matplotlib networkx docopt
   ```

## Usage

To run the script, use the following command format:

```bash
python bonus_analysis.py FILE [MODE] [COLUMN] [--exclude=<players>]
```

### Arguments and Options

- `FILE`: The path to the CSV file containing the bonus data.
- `MODE`: Specify the analysis mode (`prediction_freq`, `groups`, or `prediction_network`).
- `COLUMN`: The column in the CSV file to analyze (applicable only in `prediction_freq` mode).
- `--exclude=<players>`: A comma-separated list of player names to exclude from the analysis.

### Examples

In all cases, the script will output general information about the data and then display the results of the analysis.

1. Analyzing prediction frequencies for a specific column:
   ```bash
   python bonus_analysis.py data.csv prediction_freq WM --exclude=JohnDoe,JaneDoe
   ```
2. Generating a network graph of team predictions:
   ```bash
   python bonus_analysis.py data.csv prediction_network
   ```
3. Viewing group winner predictions:
   ```bash
   python bonus_analysis.py data.csv groups
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your proposed changes.

## License

You can find the license information in the [`LICENSE`](/license) file.

For more information, please refer to the documentation included with the script or contact the maintainers.