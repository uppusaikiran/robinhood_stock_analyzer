import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas as pd
import argparse
from tabulate import tabulate
from collections import deque
from two_factor_log_in import *
import os

import pyotp
from dotenv import load_dotenv

load_dotenv()


class StockAnalyzer:
    def __init__(self, csv_file_path, generate_data):
        # Load the CSV file
        if generate_data:
            totp = pyotp.TOTP(os.environ["robin_mfa"]).now()
            r.login(
                os.environ["robin_username"],
                os.environ["robin_password"],
                store_session=False,
                mfa_code=totp,
            )
            r.export_completed_stock_orders(".", file_name="stock_orders.csv")
            r.logout()
        print("Loading CSV file...")
        self.df = pd.read_csv(csv_file_path)
        print("CSV file loaded.")

        # Convert 'date' column to datetime format, handling mixed formats
        self.df["date"] = self.df["date"].apply(self.parse_dates)
        # Ensure 'quantity' is treated as a float with 8 decimal places
        self.df["quantity"] = self.df["quantity"].astype(float).round(8)
        print("Data preparation complete.")

    def get_current_price(self, stock_symbol):
        """
        Fetches the current price of a stock from Yahoo Finance.
        """
        try:
            stock = yf.Ticker(stock_symbol)
            price = stock.history(period="1d")["Close"].iloc[-1]
            return price
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None

    def parse_dates(self, date_str):
        """
        Parses date strings in ISO 8601 format, with or without microseconds.
        """
        try:
            # Try parsing with microseconds
            return pd.to_datetime(date_str, format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True)
        except ValueError:
            try:
                # Try parsing without microseconds
                return pd.to_datetime(date_str, format="%Y-%m-%dT%H:%M:%SZ", utc=True)
            except ValueError:
                # If all else fails, return NaT
                print(f"Failed to parse date: {date_str}")
                return pd.NaT

    def filter_stock_transactions(self, stock_symbol):
        """
        Filters transactions for a particular stock symbol and prints them in a tabular format.
        """
        print(f"Filtering transactions for stock symbol: {stock_symbol}")
        # Filter by stock symbol
        stock_data = self.df[self.df["symbol"] == stock_symbol.upper()]
        print(f"Found {len(stock_data)} transactions for {stock_symbol}.")

        if stock_data.empty:
            print(f"No transactions found for stock symbol: {stock_symbol}")
            return

        # Prepare data for tabulate
        table_data = stock_data[["date", "symbol", "side", "quantity", "average_price"]]

        # Print table
        headers = ["Date", "Symbol", "Side", "Quantity", "Average Price"]
        print(f"Transactions for {stock_symbol}:")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    def analyze_stock_totals(self, stock_symbol, start_date=None):
        """
        Analyzes total bought and sold quantities for the specified stock.
        """
        print(f"Analyzing totals for stock symbol: {stock_symbol}")
        # Filter by stock symbol
        stock_data = self.df[self.df["symbol"] == stock_symbol.upper()]

        if start_date:
            start_date = pd.to_datetime(start_date)
            stock_data = stock_data[stock_data["date"] >= start_date]
            print(f"Filtering data from start date: {start_date.date()}")

        if stock_data.empty:
            print(
                f"No transactions found for stock symbol: {stock_symbol} with the specified date filter."
            )
            return

        # Calculate the total bought and sold quantities
        total_bought = stock_data[stock_data["side"] == "buy"]["quantity"].sum()
        total_sold = stock_data[stock_data["side"] == "sell"]["quantity"].sum()

        # Print the results
        print(f"Total analysis for {stock_symbol}:")
        if start_date:
            print(f"Transactions from {start_date.date()} to today:")
        print(f"Total Bought Quantity: {total_bought:.8f}")
        print(f"Total Sold Quantity: {total_sold:.8f}")

    def calculate_outstanding_stock(self, stock_symbol, todays_price):
        """
        Calculates the outstanding stock count for a particular stock symbol
        by matching buy and sell transactions in FIFO order and displays detailed information.

        :param stock_symbol: The stock symbol to analyze.
        :param todays_price: Today's price of the stock.
        """
        print(f"Calculating outstanding stock for symbol: {stock_symbol}")

        # Filter by stock symbol and sort by date
        stock_data = self.df[self.df["symbol"] == stock_symbol.upper()]
        stock_data = stock_data.sort_values(by="date")

        # Ensure today's date is timezone-aware
        today = pd.Timestamp.now(tz="UTC")

        # Separate buys and sells
        buys = stock_data[stock_data["side"] == "buy"]
        sells = stock_data[stock_data["side"] == "sell"]

        outstanding_stocks = deque()  # FIFO queue for outstanding stocks
        detailed_sales = []

        # Process buys
        for _, buy in buys.iterrows():
            outstanding_stocks.append(
                (buy["date"], buy["quantity"], buy["average_price"])
            )

        # Process sells
        for _, sell in sells.iterrows():
            sell_quantity = sell["quantity"]
            while sell_quantity > 0 and outstanding_stocks:
                buy_date, buy_quantity, buy_price = outstanding_stocks.popleft()
                if buy_quantity > sell_quantity:
                    # Partial match
                    detailed_sales.append(
                        [
                            sell["date"],
                            "Sell",
                            f"{sell_quantity:.8f}",
                            buy_date,
                            f"{buy_quantity:.8f}",
                            "Partial",
                            (
                                "Short Term"
                                if (sell["date"] - buy_date).days <= 365
                                else "Long Term"
                            ),
                        ]
                    )
                    outstanding_stocks.appendleft(
                        (buy_date, buy_quantity - sell_quantity, buy_price)
                    )
                    sell_quantity = 0
                else:
                    # Fully matched
                    detailed_sales.append(
                        [
                            sell["date"],
                            "Sell",
                            f"{buy_quantity:.8f}",
                            buy_date,
                            f"{buy_quantity:.8f}",
                            "Full",
                            (
                                "Short Term"
                                if (sell["date"] - buy_date).days <= 365
                                else "Long Term"
                            ),
                        ]
                    )
                    sell_quantity -= buy_quantity

        # Calculate remaining outstanding quantity
        outstanding_stocks_info = []
        outstanding_quantity = 0

        total_cost_value = 0
        total_current_value = 0
        total_profit_loss = 0

        for date, quantity, avg_price in outstanding_stocks:
            # Calculate current value, cost value, and profit/loss
            current_value = quantity * todays_price
            cost_value = quantity * avg_price
            profit_loss = current_value - cost_value
            holding_period = "Long Term" if (today - date).days > 365 else "Short Term"
            profit_loss_colored = (
                f"\033[92m{profit_loss:.8f}\033[0m"
                if profit_loss >= 0
                else f"\033[91m{profit_loss:.8f}\033[0m"
            )
            holding_period_colored = (
                f"\033[92m{holding_period}\033[0m"
                if holding_period == "Long Term"
                else f"\033[91m{holding_period}\033[0m"
            )

            outstanding_stocks_info.append(
                [
                    date,
                    f"{quantity:.8f}",
                    f"{avg_price:.8f}",
                    f"{todays_price:.8f}",
                    f"{current_value:.8f}",
                    profit_loss_colored,
                    holding_period_colored,
                ]
            )
            outstanding_quantity += quantity
            total_cost_value += cost_value
            total_current_value += current_value
            total_profit_loss += profit_loss

        # Add a row for totals
        totals_row = [
            "Totals",
            f"{outstanding_quantity:.8f}",
            "",
            "",
            f"{total_current_value:.8f}",
            f"{total_profit_loss:.8f}",
            "",
        ]
        outstanding_stocks_info.append(totals_row)

        # Print detailed sales table
        headers = [
            "Sale Date",
            "Transaction Type",
            "Sale Quantity",
            "Buy Date",
            "Buy Quantity",
            "Match Type",
            "Holding Period",
        ]
        print("Detailed Sales Information:")
        print(tabulate(detailed_sales, headers=headers, tablefmt="fancy_grid"))

        # Print remaining outstanding stocks
        outstanding_headers = [
            "Buy Date",
            "Outstanding Quantity",
            "Average Purchase Price",
            "Today's Price",
            "Current Value",
            "Profit/Loss",
            "Holding Period",
        ]
        print("\nOutstanding Stocks:")
        print(
            tabulate(
                outstanding_stocks_info,
                headers=outstanding_headers,
                tablefmt="fancy_grid",
            )
        )

        # Print the final outstanding quantity
        print(
            f"\nOutstanding stock count for {stock_symbol}: {outstanding_quantity:.8f}"
        )

        # Plot the historical prices and transactions
        self.plot_stock_analysis(stock_symbol)

    def plot_stock_analysis(self, stock_symbol):
        """
        Plots the historical price and transactions, highlighting key points.
        """
        # Fetch historical stock data from Yahoo Finance
        stock = yf.Ticker(stock_symbol)
        historical_data = stock.history(period="1y")
        historical_data.reset_index(inplace=True)
        historical_data["Date"] = pd.to_datetime(historical_data["Date"])

        # Find local min and max
        local_minima = signal.argrelextrema(
            historical_data["Close"].values, np.less_equal, order=10
        )[0]
        local_maxima = signal.argrelextrema(
            historical_data["Close"].values, np.greater_equal, order=10
        )[0]
        global_min = np.argmin(historical_data["Close"].values)
        global_max = np.argmax(historical_data["Close"].values)

        # Convert Timestamps to datetime for comparison
        historical_data["Date"] = historical_data["Date"].dt.to_pydatetime()
        local_min_dates = historical_data.iloc[local_minima]["Date"]
        local_max_dates = historical_data.iloc[local_maxima]["Date"]
        global_min_date = historical_data.iloc[global_min]["Date"]
        global_max_date = historical_data.iloc[global_max]["Date"]

        # Plot historical prices
        plt.figure(figsize=(14, 8))
        plt.plot(
            historical_data["Date"],
            historical_data["Close"],
            label="Historical Price",
            color="blue",
            linestyle="dashed",
        )

        # Highlight local and global minima and maxima
        plt.scatter(
            local_min_dates,
            historical_data.loc[local_minima, "Close"],
            color="green",
            label="Local Minima",
            zorder=5,
        )
        plt.scatter(
            local_max_dates,
            historical_data.loc[local_maxima, "Close"],
            color="red",
            label="Local Maxima",
            zorder=5,
        )
        plt.scatter(
            global_min_date,
            historical_data.loc[global_min, "Close"],
            color="purple",
            label="Global Min",
            zorder=5,
        )
        plt.scatter(
            global_max_date,
            historical_data.loc[global_max, "Close"],
            color="orange",
            label="Global Max",
            zorder=5,
        )

        # Highlight key points
        transactions = self.df[self.df["symbol"] == stock_symbol.upper()]
        transactions["date"] = transactions["date"].dt.to_pydatetime()
        buy_dates = pd.Series(
            transactions[transactions["side"] == "buy"]["date"].values
        )
        sell_dates = pd.Series(
            transactions[transactions["side"] == "sell"]["date"].values
        )
        buy_prices = transactions[transactions["side"] == "buy"]["average_price"].values
        sell_prices = transactions[transactions["side"] == "sell"][
            "average_price"
        ].values

        # Determine good and bad transactions
        good_buy_mask = buy_dates.isin(local_min_dates) | buy_dates.isin(
            [global_min_date]
        )
        good_sell_mask = sell_dates.isin(local_max_dates) | sell_dates.isin(
            [global_max_date]
        )
        bad_buy_mask = buy_dates.isin(local_max_dates) | buy_dates.isin(
            [global_max_date]
        )
        bad_sell_mask = sell_dates.isin(local_min_dates) | sell_dates.isin(
            [global_min_date]
        )

        # Plot buy and sell transactions as lines
        plt.plot(
            buy_dates,
            buy_prices,
            color="darkgreen",
            linestyle="-",
            zorder=5,
        )
        plt.plot(
            sell_dates,
            sell_prices,
            color="darkred",
            linestyle="-",
            zorder=5,
        )

        # Highlight good and bad conditions
        plt.scatter(
            buy_dates[good_buy_mask].values,
            buy_prices[good_buy_mask],
            color="lime",
            edgecolor="black",
            s=100,
            label="Good Buy",
            zorder=6,
        )
        plt.scatter(
            buy_dates[bad_buy_mask].values,
            buy_prices[bad_buy_mask],
            color="red",
            edgecolor="black",
            s=100,
            label="Bad Buy",
            zorder=6,
        )
        plt.scatter(
            sell_dates[good_sell_mask].values,
            sell_prices[good_sell_mask],
            color="orange",
            edgecolor="black",
            s=100,
            label="Good Sell",
            zorder=6,
        )
        plt.scatter(
            sell_dates[bad_sell_mask].values,
            sell_prices[bad_sell_mask],
            color="darkred",
            edgecolor="black",
            s=100,
            label="Bad Sell",
            zorder=6,
        )

        # Annotate transactions
        # for date, price in zip(buy_dates, buy_prices):
        #     plt.annotate(
        #         "Buy",
        #         (date, price),
        #         textcoords="offset points",
        #         xytext=(0, 5),
        #         ha="center",
        #         color="cyan",
        #     )
        # for date, price in zip(sell_dates, sell_prices):
        #     plt.annotate(
        #         "Sell",
        #         (date, price),
        #         textcoords="offset points",
        #         xytext=(0, 5),
        #         ha="center",
        #         color="magenta",
        #     )

        # Format the plot
        plt.title(f"{stock_symbol} - Historical Prices and Transactions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        # Move legend to the bottom
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

        # Show plot
        plt.show()


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze stock transactions from a CSV file."
    )
    parser.add_argument("symbol", type=str, help="The stock symbol to analyze")
    parser.add_argument(
        "--start_date",
        type=str,
        help="The start date (YYYY-MM-DD) to analyze from",
        default=None,
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        help="The path to the CSV file",
        default="./stock_orders.csv",
    )
    parser.add_argument("--generate_data", type=str, required=False, default=False)

    args = parser.parse_args()

    # Create an instance of StockAnalyzer
    analyzer = StockAnalyzer(args.file, args.generate_data)

    # Filter and print transactions
    analyzer.filter_stock_transactions(args.symbol)

    # Analyze and print total bought and sold quantities
    analyzer.analyze_stock_totals(args.symbol, start_date=args.start_date)

    # Fetch the current price and calculate outstanding stocks
    current_price = analyzer.get_current_price(args.symbol)
    if current_price is not None:
        analyzer.calculate_outstanding_stock(args.symbol, current_price)
    else:
        print(f"Could not fetch the current price for {args.symbol}")


if __name__ == "__main__":
    main()
