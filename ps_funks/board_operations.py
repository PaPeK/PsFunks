import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
from sklearn import metrics
from sklearn.model_selection import train_test_split
from ps_funks.translate import translate_auto_2_en


def load_and_binarize_and_translate(f_name, translate_by="api"):
    if translate_by == "google":
        translator = GoogleTranslator(source="auto", target="en")
        trans_func = translator.translate
    elif translate_by == "api":
        trans_func = translate_auto_2_en
    df = (
        pd.read_csv(f_name)
        .loc[:, ["title", "sentiment"]]
        .assign(
            sentiment=lambda x: np.where(x["sentiment"] == "Noise", 0, 1),
            title=lambda x: x["title"].apply(trans_func),
        )
    )
    return df


class BoardData:
    """
    handles data from different boards
    - loads data
    - translates to english
    - create test and train data
    """

    def __init__(self, test_size=0.33, random_state=42):
        self._train_test_split_paras = dict(test_size=test_size, random_state=random_state)
        self.board_names = []
        self.board_names_combined = []
        self._data = {}

    def add_board(self, f_name, board_name, df=None):
        assert (
            board_name not in self.board_names
        ), f"board_name {board_name} already in {self.board_names}"
        if df is None:
            df = load_and_binarize_and_translate(f_name, translate_by="api")
        x_train, x_test, y_train, y_test = train_test_split(
            df["title"],
            df["sentiment"],
            stratify=df["sentiment"],
            **self._train_test_split_paras,
        )
        board_data = {
            "df": df,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        self._data[board_name] = board_data
        self.board_names.append(board_name)

    def get_test_xy(self, board_name):
        if type(board_name) is list:
            board_name = self._handle_multiple_board_names(board_name)
        dat = self._data[board_name]
        x, y = [
            dat["x_test"],
            dat["y_test"],
        ]
        return x, y, f"{board_name}_test_data"

    def get_train_xy(self, board_name):
        if type(board_name) is list:
            board_name = self._handle_multiple_board_names(board_name)
        dat = self._data[board_name]
        x, y = [
            dat["x_train"],
            dat["y_train"],
        ]
        return x, y, f"{board_name}_train_data"

    def get_strat_xy(self, board_name):
        if type(board_name) is list:
            board_name = self._handle_multiple_board_names(board_name)
        dat = self._data[board_name]
        x, y = [
            pd.concat([dat["x_train"], dat["x_test"]]),
            pd.concat([dat["y_train"], dat["y_test"]]),
        ]
        return x, y, f"{board_name}_strat_data"

    def combine_boards(self, board_names):
        board_name = self._handle_multiple_board_names(board_names)
        print(board_name, " created")

    def _handle_multiple_board_names(self, board_names):
        board_name = "_AND_".join(board_names)
        print("creating muliboard: ", board_name)
        if not np.all(np.isin(board_names, self.board_names)):
            missing_boards = np.array(board_names)[~np.isin(board_names, self.board_names)]
            raise ValueError(
                f"board_name {missing_boards} not in {self.board_names} --> load it via add_board"
            )
        if board_name not in self.board_names:
            df = pd.concat([self._data[b_name]["df"] for b_name in board_names])
            self.add_board(None, board_name, df=df)
            self.board_names_combined.append(board_name)
        return board_name

    def get_board_statistics(self):
        dat = []
        for bn in self.board_names:
            df = self._data[bn]["df"]
            dat.append(np.bincount(df.sentiment))
        df = pd.DataFrame(dat, columns=["Noise", "Not-Noise"], index=self.board_names)
        df["Noise_fraction"] = df.Noise / df.sum(axis=1)
        return df


def print_test_prediction(model, x_test, y_test):
    """
    prints the metrids of the test prediction
    """
    predicted = model.predict(x_test)
    print(
        metrics.classification_report(
            y_test,
            predicted,
            labels=[0, 1],
            target_names=["Noise", "Not-Noise"],
        )
    )


def test_prediction_df(model, x_test, y_test, multi_index=None):
    """
    returns the prediction metrics of the prediction as a dataframe
        if multi_index is not none -> multi_index used as highest level of column-multindex
        +-----------+-------------------------------+-----------------------------------+
        |           |   ('board_7031_oct', 'Noise') |   ('board_7031_oct', 'Not-Noise') |
        |-----------+-------------------------------+-----------------------------------|
        | Precision |                      0.672414 |                          0.595238 |
        | Recall    |                      0.696429 |                          0.568182 |
        | F-Score   |                      0.684211 |                          0.581395 |
        | Support   |                     56        |                         44        |
        +-----------+-------------------------------+-----------------------------------+
    """
    predicted = model.predict(x_test)
    metrics.classification_report(
        y_test,
        predicted,
        labels=[0, 1],
        target_names=["Noise", "Not-Noise"],
    )
    out = metrics.precision_recall_fscore_support(
        y_test,
        predicted,
        labels=[0, 1],
    )
    df = pd.DataFrame(
        out, columns=["Noise", "Not-Noise"], index=["Precision", "Recall", "F-Score", "Support"]
    )
    if multi_index is not None:
        df.columns = pd.MultiIndex.from_product([[multi_index], df.columns])
    return df
