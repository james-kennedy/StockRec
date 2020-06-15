import logging
import json
import os

import pandas as pd
from flask import Flask, redirect, render_template, request
from gevent import pywsgi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


logging.basicConfig(level=logging.INFO)

# Input stock data path. Assumes format from fetch_data.py
INPUT_DATA = os.path.join(os.getcwd(), "spy_data.json")

# Vars for flask web app
HOST = "127.0.0.1"
PORT = 5000
app = Flask(__name__)


class RecommendationSystem:
    def __init__(self):
        """
        The constructor for the Recommendation System. Update field names here to include/exclude more.
        """
        # some manual config for which fields we want to limit to (original data is 100+)
        self.field_lookup = {
            "Name": "shortName",
            "Ticker": "symbol",
            "Number of Employees": "fullTimeEmployees",
            "Business Summary": "longBusinessSummary",
            "Industry": "industry",
            "Market Open": "regularMarketOpen",
            "200 Day Ave.": "twoHundredDayAverage",
            "Payout Ratio": "payoutRatio",
            "Trailing Annual Div Rate": "trailingAnnualDividendRate",
            "50 Day Ave.": "fiftyDayAverage",
            "Trailing P/E": "trailingPE",
            "Market Cap": "marketCap",
            "Ave. Volume": "averageVolume",
            "Price to Sales 12 Months": "priceToSalesTrailing12Months",
            "52 Week High": "fiftyTwoWeekHigh",
            "52 Week Low": "fiftyTwoWeekLow",
            "Profit Margins": "profitMargins",
            "PEG Ratio": "pegRatio"
        }
        self.exclude_from_fundamentals = [
            "shortName", 
            "symbol", 
            "longBusinessSummary", 
            "industry"
        ]
        self.inverse_field_lookup = {v: k for k, v in self.field_lookup.items()}
        self.useful_fields = [v for k, v in self.field_lookup.items()]
        self.fundamentals = [self.inverse_field_lookup[x] for x in self.useful_fields 
            if x not in self.exclude_from_fundamentals]

        # load in and clean the data
        pd_all_data = self.load_pandas_df_from_json()
        self.stock_data, self.indices_map = self.clean_data_frame(pd_all_data)
        self.all_tickers = self.indices_map.keys().tolist()

        logging.info("Data loaded")

        # use NLP to get similarities between business summaries
        self.cosine_sim_tfidf = self.business_summary_sim()

        # placeholder for sim matrix
        self.sim_matrix = None

        logging.info("App ready")

    def rebuild_sim_matrix(self, columns):
        """
        Recalculate the similarity matrix based on the input columns

        Parameters:
            columns (list of string):   columns of the dataframe
        """
        # create multidimensional space from numeric fundamentals and use that for similarity
        self.sim_matrix = pairwise_distances(self.stock_data[columns], metric="mahalanobis")

    def business_summary_sim(self):
        """
        Calculate cosine similarity of the long business summaries between stocks
        Use tfidf to lower importance of words that appear frequently in business summaries
        """
        tfidf = TfidfVectorizer(stop_words='english')       # removes English stop words
        tfidf_matrix = tfidf.fit_transform(self.stock_data['longBusinessSummary'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    def get_recommendations(self, ticker, cosine_sim, reverse_scores=True):
        """
        Given a ticker, use a similarity matrix to find the 10 closest other stocks

        Paramters:
            ticker (string):            ticker string
            cosine_sim (sim matrix):    similarity matrix for all tickers
            reverse_scores (bool):      reverse scores if lower is closer

        Return:                
            (df, list of float):        dataframe of symbols and shortName, list of scores
        """
        idx = self.indices_map[ticker]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=reverse_scores)
        sim_scores = sim_scores[1:11]   # take top 10
        similar_tickers = [i[0] for i in sim_scores]
        score_list = [(1 - i[1]) for i in sim_scores]
        if reverse_scores:
            score_list = [(1 - x) for x in score_list]
        return self.stock_data[["symbol", "shortName"]].iloc[similar_tickers], score_list

    def get_recommendations_single_field(self, ticker, field):
        """
        Given a ticker and a field of interest, find the absolute 10 closest other stocks on this value

        Parameters:
            ticker (string):            ticker string
            field (string):             field to sort on
        
        Return:
            (df, list of float):        dataframe of symbols and shortName, list of scores
        """
        idx = self.indices_map[ticker]
        this_value = self.stock_data[field].iloc[idx]
        # sort absolute values for this column against the input ticker
        closest_values = self.stock_data.iloc[(self.stock_data[field] - this_value).abs().argsort()[1:11]]
        scores = [(this_value - x) for x in closest_values[field]]
        return closest_values[["symbol", "shortName"]], scores

    def load_pandas_df_from_json(self, json_file_path=INPUT_DATA):
        """
        Load a .json dictionary object with stock data and turn it into a pandas df

        Parameters:
            json_file_path (string):    string to .json file with data

        Return:
            (pandas df)                 json data converted into a df
        """
        with open(json_file_path, "r") as input_file:
            json_data = json.loads(input_file.read())
        assert "stock_data" in json_data
        return pd.DataFrame(json_data["stock_data"]).transpose()

    def clean_data_frame(self, pd_df):
        """
        Perform cleaning operations on the dataframe. 
        Note that using the current implementation, this removes rows if cells have missing information.
        To be more inclusive, some fields could be left out, or default values could be entered for the missing ones.
        However, using default values will come at the cost of a less accurate similarity calculation.

        Parameters:
            pd_df (pandas df):          input stock data with all fields from yfinance

        Return:  
            (pandas df, indices map):   filtered data, indices map from ticker to index
        """
        pd_df["longBusinessSummary"] = pd_df["longBusinessSummary"].fillna("")      # replace NaN with empty string
        pd_df = pd_df[self.useful_fields]                                           # remove fields we don't care about
        pd_df = pd_df.dropna()                                                      # remove any rows with missing data
        pd_df = pd_df.reset_index()                                                 # turn index to int base 0
        indices = pd.Series(pd_df.index, index=pd_df['symbol']).drop_duplicates()   # make index to ticker map
        return pd_df, indices


@app.route("/get_recommendation", methods=["GET", "POST"])
def page_get_recommendation():
    if request.method == "POST":
        ticker = request.form["tickers"]
        option = request.form['options']
        if option == "biz_summary":
            option_text = "business summary"
            recommendations, scores = StockRec.get_recommendations(ticker, StockRec.cosine_sim_tfidf)
        else:
            factors = request.form.getlist('factors')
            option_text = "stock fundamentals ( {} )".format(", ".join(factors))
            field_names = [StockRec.field_lookup[x] for x in factors]   # map from human-readable to field names
            # Check how many fields we are comparing on - need at least 1, 1 uses sorting, 2+ uses sim matrix
            if len(field_names) == 0:
                logging.info("No fields given for fundamentals similarity recommendation")
                return render_template(
                    "index.html", 
                    tickers=StockRec.all_tickers,
                    fields=StockRec.fundamentals,
                    error_msg="Select at least 1 field to compare on stock fundamentals"
                )
            elif len(field_names) == 1:
                logging.info("Single field given for fundamentals similarity recommendation")
                recommendations, scores = StockRec.get_recommendations_single_field(ticker, field_names[0])
            else:
                logging.info("Multiple fields given for fundamentals similarity recommendation")
                StockRec.rebuild_sim_matrix(field_names)
                recommendations, scores = StockRec.get_recommendations(ticker, StockRec.sim_matrix, False)
        logging.info("Fetched recommendation for: {}".format(ticker))
        recommendations = recommendations.to_dict("records")
        for rec, score in zip(recommendations, scores):
            rec["score"] = round(score, 2)
    return render_template(
        "index.html", 
        tickers=StockRec.all_tickers,
        fields=StockRec.fundamentals,
        recs=recommendations, 
        cols=["symbol", "shortName", "score"],
        display_cols=["Ticker", "Name", "Score"],
        input_ticker=ticker,
        by_similarity=option_text
    )


@app.route("/")
def index():
    return render_template(
        "index.html", 
        tickers=StockRec.all_tickers,
        fields=StockRec.fundamentals
    )


def main():
    global StockRec, app
    StockRec = RecommendationSystem()
    logging.info("App launching at http://{}:{}".format(HOST, PORT))
    server = pywsgi.WSGIServer((HOST, PORT), app)
    server.serve_forever() 


if __name__ == "__main__":
    main()
