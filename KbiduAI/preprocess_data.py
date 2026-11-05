# preprocess_data.py
import os
import argparse
from data_preprocessor import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess training data for Kbidu AI.")
    parser.add_argument("--input", required=True, help="Path to raw input file")
    parser.add_argument("--output", required=True, help="Path to cleaned output file")
    parser.add_argument("--min_words", type=int, default=5, help="Minimum words per sample")
    args = parser.parse_args()

    preprocessor = DataPreprocessor()
    preprocessor.preprocess_file(args.input, args.output)

if __name__ == "__main__":
    main()