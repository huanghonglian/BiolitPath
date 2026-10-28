#!/usr/bin/env python3
"""Evaluate PubTator predictions on the word-level BIO data used in training."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from seqeval.metrics.sequence_labeling import get_entities


DATASETS = {
    "ncbi-disease": "NCBI-disease",
    "BC4CHEMD": "BC4CHEMD",
    "Revised_JNLPBA-ct": "Revised_JNLPBA-ct",
    "CAFETERIA": "CAFETERIA",
    "CONLL2003": "CONLL2003",
    "CRAFT-go": "CRAFT-go",
    "Medmention-diagnosis": "Medmention-diagnose",
    "Medmention-treatment": "Medmention-treatment",
    "NERO-bp": "NERO-bp",
}

ENTITY_TYPES = {
    "ncbi-disease": "disease",
    "BC4CHEMD": "chemical",
    "Revised_JNLPBA-ct": "celltype",
    "CAFETERIA": "food",
    "CONLL2003": "location",
    "CRAFT-go": "go",
    "Medmention-diagnosis": "diagnosis",
    "Medmention-treatment": "treatment",
    "NERO-bp": "bodypart",
}

GOLD_NAMES = {"Medmention-diagnosis": "Medmention-diagnose"}

MODEL_NAMES = {
    "Gold-only_Baseline": "Gold-only Baseline",
    "BiolitNER": "BiolitNER",
    "BERN2": "BERN2",
    "HunFlair2": "HunFlair2",
}



@dataclass
class PubTatorDocument:
    doc_id: str
    text: str
    entities: list[tuple[int, int, str]]


def read_bio(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    all_tokens, all_labels = [], []
    tokens, labels = [], []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    all_tokens.append(tokens)
                    all_labels.append(labels)
                    tokens, labels = [], []
                continue

            parts = line.split(" ")
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            labels.append(parts[-1].strip())

    if tokens:
        all_tokens.append(tokens)
        all_labels.append(labels)

    return all_tokens, all_labels


def read_pubtator(path: Path) -> list[PubTatorDocument]:
    documents = []
    content = path.read_text(encoding="utf-8").strip()

    for block in content.split("\n\n"):
        doc_id = None
        title = ""
        abstract = ""
        entities = []

        for line in block.splitlines():
            if "|t|" in line:
                line_id, title = line.split("|t|", 1)
                doc_id = doc_id or line_id
                if line_id != doc_id:
                    raise ValueError(f"{path}: inconsistent document IDs in one block")
            elif "|a|" in line:
                line_id, abstract = line.split("|a|", 1)
                doc_id = doc_id or line_id
                if line_id != doc_id:
                    raise ValueError(f"{path}: inconsistent document IDs in one block")
            elif "\t" in line:
                fields = line.split("\t")
                if len(fields) < 4:
                    raise ValueError(f"{path}: malformed annotation: {line}")
                entity_id, start, end, mention = fields[:4]
                if doc_id is not None and entity_id != doc_id:
                    raise ValueError(f"{path}: annotation ID differs from text ID: {line}")
                entities.append((int(start), int(end), mention))

        if doc_id is None:
            raise ValueError(f"{path}: document block has no title/abstract line")

        text = title + " " + abstract
        for start, end, mention in entities:
            if not 0 <= start < end <= len(text):
                raise ValueError(f"{path}: invalid span {doc_id} [{start}, {end})")
            if text[start:end] != mention:
                raise ValueError(
                    f"{path}: {doc_id} [{start}, {end}) is {text[start:end]!r}, "
                    f"not {mention!r}"
                )

        documents.append(PubTatorDocument(doc_id, text, entities))

    ids = [document.doc_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate document IDs")

    return documents


def locate_bio_tokens(text: str, tokens: list[str], context: str) -> list[tuple[int, int]]:
    offsets = []
    position = 0

    for token in tokens:
        while position < len(text) and text[position].isspace():
            position += 1
        start = position
        end = start + len(token)
        actual = text[start:end]
        delimiter_escape = token == "|" and actual == "/"
        if actual != token and not delimiter_escape:
            raise ValueError(
                f"{context}: BIO token {token!r} does not match text at position {start}: "
                f"{actual!r}"
            )
        offsets.append((start, end))
        position = end

    if text[position:].strip():
        raise ValueError(f"{context}: PubTator text contains unmatched trailing text")

    return offsets


def prediction_to_bio(
    document: PubTatorDocument,
    offsets: list[tuple[int, int]],
) -> tuple[list[str], int, int]:
    projected_spans = set()
    unaligned_spans = set()

    for start, end, _ in document.entities:
        covered = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start
        ]
        if not covered:
            raise ValueError(
                f"{document.doc_id}: entity [{start}, {end}) overlaps no BIO token"
            )

        first, last = covered[0], covered[-1]
        if start != offsets[first][0] or end != offsets[last][1]:
            unaligned_spans.add((start, end))
            continue
        projected_spans.add((first, last))

    # Flat BIO cannot represent overlaps; deterministically keep longer spans.
    selected_spans = []
    for span in sorted(projected_spans, key=lambda item: (-(item[1] - item[0] + 1), item)):
        if any(span[0] <= kept[1] and kept[0] <= span[1] for kept in selected_spans):
            continue
        selected_spans.append(span)
    ordered_spans = sorted(selected_spans)
    dropped_overlaps = len(projected_spans) - len(ordered_spans)

    labels = ["O"] * len(offsets)
    for first, last in ordered_spans:
        labels[first] = "B"
        for index in range(first + 1, last + 1):
            labels[index] = "I"

    return labels, len(unaligned_spans), dropped_overlaps


def bio_entities(label_sequences: list[list[str]]) -> set[tuple[int, int, int]]:
    entities = set()
    for document_index, labels in enumerate(label_sequences):
        for _, start, end in get_entities(labels):
            entities.add((document_index, start, end))
    return entities


def evaluate_dataset(
    bio_path: Path,
    gold_pubtator_path: Path,
    prediction_path: Path,
) -> tuple[float, float, float, int, int]:
    tokens, gold_labels = read_bio(bio_path)
    references = read_pubtator(gold_pubtator_path)
    predictions = read_pubtator(prediction_path)

    if len(tokens) != len(references):
        raise ValueError(
            f"{bio_path}: {len(tokens)} BIO sequences but "
            f"{gold_pubtator_path} has {len(references)} documents"
        )
    if len(references) != len(predictions):
        raise ValueError(
            f"{prediction_path}: expected {len(references)} documents, "
            f"found {len(predictions)}"
        )

    predicted_labels = []
    unaligned = 0
    dropped_overlaps = 0

    for index, (words, reference, prediction) in enumerate(
        zip(tokens, references, predictions)
    ):
        if reference.doc_id != prediction.doc_id:
            raise ValueError(
                f"document {index}: ID mismatch: {reference.doc_id!r} != "
                f"{prediction.doc_id!r}"
            )
        if reference.text != prediction.text:
            raise ValueError(f"{reference.doc_id}: Gold and prediction text differ")

        offsets = locate_bio_tokens(reference.text, words, reference.doc_id)
        labels, count, dropped = prediction_to_bio(prediction, offsets)
        predicted_labels.append(labels)
        unaligned += count
        dropped_overlaps += dropped

    gold_entities = bio_entities(gold_labels)
    prediction_entities = bio_entities(predicted_labels)
    true_positives = len(gold_entities & prediction_entities)
    false_positives = (
        len(prediction_entities - gold_entities)
        + unaligned
        + dropped_overlaps
    )
    false_negatives = len(gold_entities - prediction_entities)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1, unaligned, dropped_overlaps


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate PubTator predictions using training BIO tokens and seqeval"
    )
    parser.add_argument(
        "--nerdata-dir", type=Path, default=script_dir / "Testing_BIO"
    )
    parser.add_argument("--gold-dir", type=Path, default=script_dir / "Gold_standard")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Gold-only_Baseline", "BiolitNER", "BERN2", "HunFlair2"],
    )
    parser.add_argument(
        "--output", type=Path, default=script_dir / "eval_results_training_bio.csv"
    )
    args = parser.parse_args()

    rows = []
    for dataset, bio_directory in DATASETS.items():
        bio_path = args.nerdata_dir / bio_directory / "test.txt"
        gold_name = GOLD_NAMES.get(dataset, dataset)
        gold_path = args.gold_dir / f"{gold_name}.txt"

        for model in args.models:
            prediction_path = script_dir / model / f"{dataset}.txt"
            if not prediction_path.exists():
                continue

            precision, recall, f1, unaligned, dropped_overlaps = evaluate_dataset(
                bio_path, gold_path, prediction_path
            )
            row = {
                "Dataset": dataset,
                "Type": ENTITY_TYPES[dataset],
                "Model": MODEL_NAMES.get(model, model),
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "F1": f"{f1:.3f}",
            }
            rows.append(row)
            print(
                f"{dataset:24} {MODEL_NAMES.get(model, model):22} "
                f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
                f"unaligned={unaligned}"
            )

    fieldnames = ["Dataset", "Type", "Model", "Precision", "Recall", "F1"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
