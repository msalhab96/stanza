
import argparse
import random
from stanza.models.classifiers.data import RankingDatum
from stanza.models.constituency import tree_reader
from stanza.server.parser_eval import EvaluateParser, ParseResult, ScoredTree
from stanza.utils.datasets.sentiment.process_utils import write_list

from tqdm import tqdm

def build_classifier_dataset(gold_trees, pred_trees, output_file):
    gold_len = len(gold_trees)
    if any(len(p) != gold_len for p in pred_trees):
        raise ValueError("Expected all tree lists to be the same length")

    dataset = []
    skipped = 0
    with EvaluateParser(classpath="$CLASSPATH", silent=True) as evaluator:
        for gold, preds in tqdm(zip(gold_trees, zip(*pred_trees)), total=gold_len):
            scores = [evaluator.process([ParseResult(gold, [(pred, 1.0)], None, None)]).f1 for pred in preds]
            max_score = max(scores)
            min_score = min(scores)
            if max_score == min_score:
                continue
            max_idx = random.choice([idx for idx, score in enumerate(scores) if score == max_score])
            min_idx = random.choice([idx for idx, score in enumerate(scores) if score == min_score])
            dataset.append(RankingDatum("1", gold.leaf_labels(), preds[max_idx], preds[min_idx]))

    write_list(output_file, dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, default=None, required=True, help="Gold treebank to compare against")
    parser.add_argument('--output_file', type=str, default=None, required=True, help="Where to write the results")
    parser.add_argument('input_file', type=str, nargs="+", help="Input treebanks to compare")
    args = parser.parse_args()

    gold_trees = tree_reader.read_treebank(args.gold_file)
    pred_trees = []
    for input_file in args.input_file:
        pred_trees.append(tree_reader.read_treebank(input_file))

    build_classifier_dataset(gold_trees, pred_trees, args.output_file)


if __name__ == "__main__":
    main()
