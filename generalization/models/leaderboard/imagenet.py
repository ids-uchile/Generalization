# Retrieves
# - "Table of all available classification weights" from https://pytorch.org/vision/stable/models.html
# -
#
#
# Usage:
#   python3 imagenet.py --download
#   python3 imagenet.py --top 10
#   python3 imagenet.py --filter "resnet50"

import argparse

import pandas as pd

pd.set_option("display.max_columns", None)


PYTORCH_URL = "https://pytorch.org/vision/stable/models.html"


def download_leaderboard():
    df = pd.read_html(PYTORCH_URL, extract_links="body")[1]
    # split tuple Weight into Weight and Card
    df["Card"] = df["Weight"].apply(
        lambda x: "https://pytorch.org/vision/stable/" + x[1]
    )
    df["Weight"] = df["Weight"].apply(lambda x: x[0])

    # retain str content
    for col in ["Acc@1", "Acc@5", "Params", "GFLOPS"]:
        df[col] = df[col].apply(lambda x: x[0])

    # Retain only url from Recipe
    df["Recipe"] = df["Recipe"].apply(lambda x: x[1])

    df.to_csv("imagenet_leaderboard.csv", index=False)


def get_leaderboard():
    try:
        df = pd.read_csv("imagenet_leaderboard.csv")
    except FileNotFoundError:
        download_leaderboard()
        df = pd.read_csv("imagenet_leaderboard.csv")
    return df


def filter(df, model_name):
    return df[df["Weight"].str.lower().str.contains(model_name.lower())]


def top(df, n):
    return df.sort_values(by=["Acc@1"], ascending=False).head(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--top", type=int)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--output", "-o", type=str, default=False)
    args = parser.parse_args()

    if args.download:
        download_leaderboard()

    df = get_leaderboard()
    if args.top:
        df = top(df, args.top)
    if args.filter:
        df = filter(df, args.filter)

    if args.output:
        df.to_csv(args.output, index=False)
    else:
        return df


if __name__ == "__main__":
    df = main()
