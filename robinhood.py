from datetime import datetime
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas as pd
import argparse
from tabulate import tabulate
from collections import deque
import os
import robin_stocks.robinhood as r

import pyotp
from dotenv import load_dotenv

load_dotenv()


class StockAnalyzer:
    def __init__(self, csv_file_path, generate_data=True):
        # Load the CSV file
        self.lastPrice = None
        totp = pyotp.TOTP(os.environ["robin_mfa"]).now()
        r.login(
            os.environ["robin_username"],
            os.environ["robin_password"],
            store_session=False,
            mfa_code=totp,
        )
        if generate_data:

            r.export_completed_stock_orders(".", file_name="stock_orders.csv")

        self.df = pd.read_csv(csv_file_path)

        # Convert 'date' column to datetime format, handling mixed formats
        self.df["date"] = self.df["date"].apply(self.parse_dates)
        # Ensure 'quantity' is treated as a float with 8 decimal places
        self.df["quantity"] = self.df["quantity"].astype(float).round(8)

        # Adjust data for stock splits
        self.adjust_for_stock_splits()

    def adjust_for_stock_splits(self):
        """
        Adjusts transactions for stock splits.
        """
        print("Adjusting data for stock splits...")
        # Define the stock splits
        stock_splits = {
            "NVDA": {
                "date": pd.Timestamp("2024-06-09 22:29:00", tz="UTC"),
                "split_factor": 88.68029 / 8.868029,
            },
            "TSLA": {
                "date": pd.Timestamp("2022-08-24 22:01:00", tz="UTC"),
                "split_factor": 3.087906 / 1.029302,
            },
        }

        # Apply splits
        for symbol, split_info in stock_splits.items():
            split_date = split_info["date"]
            split_factor = split_info["split_factor"]
            mask = (self.df["symbol"] == symbol) & (self.df["date"] < split_date)
            self.df.loc[mask, "quantity"] *= split_factor
            self.df.loc[mask, "average_price"] /= split_factor

    def get_current_price(self, stock_symbol):
        """
        Fetches the current price of a stock from Yahoo Finance.
        """
        if not self.lastPrice:
            self.lastPrice = r.get_quotes(stock_symbol, "last_trade_price")
        try:
            stock = yf.Ticker(stock_symbol)
            price = stock.history(period="1d")["Close"].iloc[-1]
            return price
        except Exception as e:

            print(f"Error fetching current price: {e}")
            return self.last_price

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

    def analyze_total_transactions_from_date(self, start_date):
        """
        Analyzes total buy and sell quantities and net profit/loss from the specified date to today for all stocks.

        :param start_date: The start date (YYYY-MM-DD) to analyze from.
        """
        # Convert start_date to datetime
        start_date = pd.to_datetime(start_date, utc=True)

        # Filter data from the start date
        filtered_df = self.df[self.df["date"] >= start_date]

        if filtered_df.empty:
            print(f"No transactions found from the specified date: {start_date.date()}")
            return

        # Group by symbol and side, then sum quantities
        grouped = (
            filtered_df.groupby(["symbol", "side"])
            .agg(
                total_quantity=pd.NamedAgg(column="quantity", aggfunc="sum"),
                avg_price=pd.NamedAgg(column="average_price", aggfunc="mean"),
            )
            .reset_index()
        )

        total_buy = grouped[grouped["side"] == "buy"].copy()
        total_sell = grouped[grouped["side"] == "sell"].copy()

        # Merge buy and sell data
        merged = pd.merge(
            total_buy, total_sell, on="symbol", how="outer", suffixes=("_buy", "_sell")
        ).fillna(0)

        # Calculate net profit/loss for each stock
        merged["net_profit_loss"] = (
            merged["total_quantity_sell"] * merged["avg_price_sell"]
        ) - (merged["total_quantity_buy"] * merged["avg_price_buy"])

        # Prepare table for display
        table_data = merged[
            ["symbol", "total_quantity_buy", "total_quantity_sell", "net_profit_loss"]
        ]

        # Print the results
        headers = [
            "Symbol",
            "Total Bought Quantity",
            "Total Sold Quantity",
            "Net Profit/Loss",
        ]
        print(f"Total transactions from {start_date.date()} to today:")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    def calculate_totals(self, start_date, end_date):
        # Convert end_date to timezone-naive if necessary
        # Convert input dates to timezone-aware
        start_date = pd.to_datetime(start_date).tz_localize("UTC")
        end_date = pd.to_datetime(end_date).tz_localize("UTC")

        # Filter data for the date range
        filtered_df = self.df[
            (self.df["date"] >= pd.to_datetime(start_date))
            & (self.df["date"] <= end_date)
        ]

        # Group by stock symbol
        grouped = filtered_df.groupby("symbol")

        # Initialize totals
        total_buy_amount = 0
        total_sell_amount = 0
        total_net_profit_loss = 0

        # Dictionary to store per stock totals
        stock_totals = {}

        for symbol, group in grouped:
            buy_transactions = group[group["side"] == "buy"]
            sell_transactions = group[group["side"] == "sell"]

            buy_amount = (
                buy_transactions["quantity"] * buy_transactions["average_price"]
            ).sum()
            sell_amount = (
                sell_transactions["quantity"] * sell_transactions["average_price"]
            ).sum()

            # Store in dictionary
            stock_totals[symbol] = {
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
            }

            # Add to overall totals
            total_buy_amount += buy_amount
            total_sell_amount += sell_amount

        # Print per stock totals
        for symbol, totals in stock_totals.items():
            print(f"Stock: {symbol}")
            print(f"  Total Buy Amount: {totals['buy_amount']}")
            print(f"  Total Sell Amount: {totals['sell_amount']}")

        # Print overall totals
        print("Overall Totals:")
        print(f"  Total Buy Amount: {total_buy_amount}")
        print(f"  Total Sell Amount: {total_sell_amount}")

    def calculate_outstanding_stock(self, stock_symbol, todays_price):
        """
        Calculates the outstanding stock count for a particular stock symbol
        by matching buy and sell transactions in FIFO order and returns detailed information.

        :param stock_symbol: The stock symbol to analyze.
        :param todays_price: Today's price of the stock.
        :return: A tuple containing (detailed_sales, outstanding_stocks_info)
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
                    profit_loss = (sell_quantity * sell["average_price"]) - (
                        sell_quantity * buy_price
                    )
                    detailed_sales.append(
                        [
                            sell["date"],
                            "Sell",
                            f"{sell_quantity:.2f}",
                            buy_date,
                            f"{buy_quantity:.2f}",
                            f"{buy_price:.2f}",
                            f"{sell['average_price']:.2f}",
                            "Partial",
                            f"{profit_loss:.2f}",
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
                    profit_loss = (buy_quantity * sell["average_price"]) - (
                        buy_quantity * buy_price
                    )
                    detailed_sales.append(
                        [
                            sell["date"],
                            "Sell",
                            f"{buy_quantity:.8f}",
                            buy_date,
                            f"{buy_quantity:.8f}",
                            f"{buy_price:.2f}",
                            f"{sell['average_price']:.2f}",
                            "Full",
                            f"{profit_loss:.2f}",
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
        total_net_profit_loss = 0

        for date, quantity, avg_price in outstanding_stocks:
            # Calculate current value, cost value, and profit/loss
            current_value = quantity * todays_price
            cost_value = quantity * avg_price
            profit_loss = current_value - cost_value
            holding_period = "Long Term" if (today - date).days > 365 else "Short Term"

            # Determine tax rate based on holding period
            tax_rate = 0.15 if holding_period == "Long Term" else 0.24
            if profit_loss > 0:
                tax = profit_loss * tax_rate
            else:
                tax = profit_loss

            # Calculate net profit/loss after tax
            if profit_loss == tax:
                net_profit_loss = profit_loss
            else:
                net_profit_loss = profit_loss - tax

            # Formatting for display
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
                    f"{avg_price:.2f}",
                    f"{todays_price:.2f}",
                    f"{current_value:.2f}",
                    f"{profit_loss:.2f}",
                    holding_period,
                    f"{tax:.2f}",
                    f"{net_profit_loss:.2f}",
                ]
            )
            outstanding_quantity += quantity
            total_cost_value += cost_value
            total_current_value += current_value
            total_profit_loss += profit_loss
            total_net_profit_loss += net_profit_loss

        # Add a row for totals
        totals_row = [
            "Totals",
            f"{outstanding_quantity:.2f}",
            "",
            "",
            f"{total_current_value:.2f}",
            f"{total_profit_loss:.2f}",
            "",
            "",
            f"{total_net_profit_loss:.2f}",
        ]
        outstanding_stocks_info.append(totals_row)

        # Return detailed sales and outstanding stocks info
        return detailed_sales, outstanding_stocks_info

    def plot_stock_analysis(self, stock_symbol):
        """
        Plots the historical price and transactions, highlighting key points.
        """
        # Filter to get the first buy date
        first_buy_date = self.df[
            (self.df["symbol"] == stock_symbol.upper()) & (self.df["side"] == "buy")
        ]["date"].min()

        if pd.isna(first_buy_date):
            print(
                f"No buy transactions found for {stock_symbol}. Cannot plot historical data."
            )
            return
        first_buy_date = first_buy_date - pd.Timedelta(days=30)
        # Calculate the period from the first buy date until today
        today = pd.Timestamp.now(tz="UTC")
        period = f"{(today - first_buy_date).days}d"

        # Fetch historical stock data from Yahoo Finance
        stock = yf.Ticker(stock_symbol)
        historical_data = stock.history(start=first_buy_date, end=today)
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

    def generate_report_data(self, stock_symbol):
        """
        Generates data for the HTML report.
        """
        # Fetch transactions and outstanding stocks
        transactions_data = self.df[self.df["symbol"] == stock_symbol.upper()]
        transactions_data = transactions_data.sort_values(by="date")

        # Calculate outstanding stock
        current_price = self.get_current_price(stock_symbol)
        if current_price is None:
            print(f"Could not fetch the current price for {stock_symbol}")
            return None

        outstanding_data = self.calculate_outstanding_stock(stock_symbol, current_price)

        # Return data as a dictionary
        return {
            "transactions": transactions_data,
            "outstanding_stocks": outstanding_data[1],  # Second item in tuple
            "sale_transactions": outstanding_data[0],  # First item in tuple
        }

    def render_html_report(self, stock_symbol, output_file):
        """
        Renders an HTML report using Jinja2 and saves it to a file.
        """
        # Generate report data
        report_data = self.generate_report_data(stock_symbol)
        if report_data is None:
            return

        # Load Jinja2 template
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("report_template.html")

        # Render HTML
        html_content = template.render(
            stock_symbol=stock_symbol,
            transactions=report_data["transactions"].to_dict(orient="records"),
            outstanding_stocks=report_data["outstanding_stocks"],
            sale_transactions=report_data["sale_transactions"],
        )

        # Save to file
        with open(output_file, "w") as f:
            f.write(html_content)

        print(f"Report saved to {output_file}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze stock transactions from a CSV file."
    )
    parser.add_argument(
        "--symbol", type=str, help="The stock symbol to analyze", required=False
    )
    parser.add_argument(
        "--spend_from",
        type=str,
        help="The start date (YYYY-MM-DD) to analyze from",
        default=None,
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="The start date (YYYY-MM-DD) to analyze from",
        default="2000-01-01",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        help="The path to the CSV file",
        default="./stock_orders.csv",
    )
    parser.add_argument("--generate_data", type=str, required=False, default=False)
    parser.add_argument(
        "--report",
        type=str,
        help="Generate an HTML report for the stock symbol",
        default="None",
    )

    args = parser.parse_args()

    # Create an instance of StockAnalyzer
    analyzer = StockAnalyzer(args.file, args.generate_data)

    if args.spend_from:
        analyzer.analyze_total_transactions_from_date(args.spend_from)
        analyzer.calculate_totals(
            args.spend_from, datetime.today().strftime("%Y-%m-%d")
        )
        sys.exit(0)

    # Filter and print transactions
    analyzer.filter_stock_transactions(args.symbol)

    # Fetch the current price and calculate outstanding stocks
    current_price = analyzer.get_current_price(args.symbol)
    if current_price is not None:
        analyzer.calculate_outstanding_stock(args.symbol, current_price)
    else:
        print(f"Could not fetch the current price for {args.symbol}")

    if args.report:
        analyzer.render_html_report(args.symbol, f"report_{args.report}.html")
        sys.exit(0)


if __name__ == "__main__":
    main()
