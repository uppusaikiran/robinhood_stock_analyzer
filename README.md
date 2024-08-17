
---

# Robinhood Stock Analyzer

A comprehensive tool for analyzing stock transactions and portfolio performance, specifically designed for integration with Robinhood. This project provides functionalities to filter and analyze stock trades, calculate outstanding stock quantities, and visualize historical stock prices. It integrates with Yahoo Finance to fetch current stock prices and plots various metrics to evaluate stock trading strategies and portfolio performance.

## Features

- **Transaction Filtering:** Filter and display stock transactions from a CSV file in a tabular format.
- **Total Analysis:** Analyze total bought and sold quantities for specific stocks.
- **Outstanding Stock Calculation:** Calculate outstanding stock quantities using FIFO order and assess profit/loss based on current prices.
- **Historical Data Visualization:** Fetch and plot historical stock prices, highlighting key buy/sell points and trends.
- **Integration with Robinhood:** Optionally generate transaction data from Robinhood using MFA authentication.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/uppusaikiran/robinhood_stock_analyzer.git
   cd robinhood_stock_analyzer
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   Create a `.env` file in the root directory with your Robinhood credentials and MFA code:

   ```plaintext
   robin_username=your_robinhood_username
   robin_password=your_robinhood_password
   robin_mfa=your_robinhood_mfa_secret
   ```

   Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Analyzer:**

   Use command-line arguments to specify the stock symbol, start date, and CSV file path.

   ```bash
   python stock_analyzer.py --symbol AAPL --start_date 2023-01-01
   ```

   **Arguments:**
   - `--symbol`: The stock symbol to analyze (e.g., AAPL).
   - `--start_date`: The start date for the analysis in YYYY-MM-DD format (optional).
   - `--file`: The path to the CSV file (optional, default is `./stock_orders.csv`).
   - `--generate_data`: Set to `True` to generate new transaction data from Robinhood (requires environment variables).

2. **Output:**

   - Displays filtered transactions for the specified stock symbol.
   - Analyzes and prints total bought and sold quantities.
   - Calculates and prints outstanding stock quantities and detailed sales information.
   - Plots historical stock prices and key transaction points.

## Example

```bash
python stock_analyzer.py --symbol MSFT --start_date 2023-01-01 --file ./my_stock_orders.csv
```

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to customize this `README.md` further if there are any additional details or features specific to your project.
