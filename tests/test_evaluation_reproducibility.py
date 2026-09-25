import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.aggregation import soft_vote, weighted_vote
from src.evaluation.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAMES,
    DEFAULT_VIRAL_BOOST,
    EvaluationConfigError,
    parse_args,
)
from src.evaluation.metadata import model_metadata, split_metadata


class EvaluationAggregationTests(unittest.TestCase):
    def setUp(self):
        # Two models, two examples, three classes.  The second example is
        # intentionally close to the viral decision boundary.
        self.model_probs = np.array(
            [
                [[0.60, 0.30, 0.10], [0.10, 0.45, 0.45]],
                [[0.40, 0.25, 0.35], [0.05, 0.44, 0.51]],
            ],
            dtype=np.float64,
        )

    def test_soft_vote_default_is_the_plain_model_mean(self):
        expected = self.model_probs.mean(axis=0)
        expected = expected / expected.sum(axis=1, keepdims=True)

        actual = soft_vote(self.model_probs)

        np.testing.assert_allclose(actual, expected)
        # A default evaluation must not change any class by a hidden boost.
        np.testing.assert_allclose(actual[:, 2], expected[:, 2])

    def test_weighted_vote_default_is_equal_weighted_mean(self):
        expected = self.model_probs.mean(axis=0)
        expected = expected / expected.sum(axis=1, keepdims=True)

        actual = weighted_vote(self.model_probs)

        np.testing.assert_allclose(actual, expected)

    def test_viral_boost_is_only_applied_when_explicitly_requested(self):
        actual = soft_vote(self.model_probs, viral_boost=1.1)
        expected = self.model_probs.mean(axis=0)
        expected[:, 2] *= 1.1
        expected = expected / expected.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.array_equal(actual, soft_vote(self.model_probs)))

    def test_explicit_ensemble_weights_are_used(self):
        actual = weighted_vote(self.model_probs, weights=[0.25, 0.75])
        expected = 0.25 * self.model_probs[0] + 0.75 * self.model_probs[1]
        expected = expected / expected.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(actual, expected)

    def test_weight_count_must_match_model_count(self):
        with self.assertRaises(ValueError):
            weighted_vote(self.model_probs, weights=[1.0])


class EvaluationConfigurationTests(unittest.TestCase):
    def test_documented_defaults_are_neutral_and_explicit(self):
        config = parse_args([])

        self.assertEqual(config.train_csv, "data/splits/train.csv")
        self.assertEqual(config.val_csv, "data/splits/val.csv")
        self.assertEqual(config.test_csv, "data/splits/test.csv")
        self.assertEqual(config.model_names, DEFAULT_MODEL_NAMES)
        self.assertEqual(config.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(config.viral_boost, DEFAULT_VIRAL_BOOST)
        self.assertEqual(config.viral_boost, 1.0)
        self.assertIsNone(config.ensemble_weights)

    def test_json_config_is_supported_and_command_line_overrides_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "evaluation.json"
            config_path.write_text(
                json.dumps(
                    {
                        "test_csv": "fixture/test.csv",
                        "models": ["mobilenet", "resnet"],
                        "viral_boost": 1.2,
                        "ensemble_weights": [0.4, 0.6],
                    }
                )
            )

            config = parse_args(
                ["--config", str(config_path), "--viral-boost", "1.1"]
            )

        self.assertEqual(config.test_csv, "fixture/test.csv")
        self.assertEqual(config.model_names, ("mobilenet", "resnet"))
        self.assertEqual(config.ensemble_weights, (0.4, 0.6))
        self.assertEqual(config.viral_boost, 1.1)

    def test_invalid_config_values_fail_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "evaluation.json"
            config_path.write_text(json.dumps({"viral_boost": -1.0}))
            with self.assertRaises(EvaluationConfigError):
                parse_args(["--config", str(config_path)])

    def test_single_evaluator_has_documented_model_default(self):
        from src.evaluation.config import DEFAULT_SINGLE_MODEL_NAMES

        config = parse_args([], default_model_names=DEFAULT_SINGLE_MODEL_NAMES)

        self.assertEqual(config.model_names, DEFAULT_SINGLE_MODEL_NAMES)


class EvaluationMetadataTests(unittest.TestCase):
    def test_split_metadata_records_manifest_identity_and_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("encoded_label\n0\n1\n2\n2\n")

            metadata = split_metadata("test", csv_path)

        self.assertEqual(metadata["split"], "test")
        self.assertEqual(metadata["path"], str(csv_path))
        self.assertEqual(metadata["samples"], 4)
        self.assertEqual(metadata["class_counts"], {"0": 1, "1": 1, "2": 2})
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_model_metadata_records_checkpoint_path_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            checkpoint = checkpoint_dir / "mobilenet.pt"
            checkpoint.write_bytes(b"checkpoint")

            metadata = model_metadata(("mobilenet",), checkpoint_dir)

        self.assertEqual(metadata[0]["name"], "mobilenet")
        self.assertEqual(metadata[0]["path"], str(checkpoint))
        self.assertEqual(len(metadata[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
