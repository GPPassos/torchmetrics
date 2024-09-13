# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from functools import partial
from itertools import product
from typing import Sequence

import pytest
from torch import Tensor
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text.bert import BERTScore
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
from typing_extensions import Literal

from unittests._helpers import skip_on_connection_issues
from unittests.text._helpers import TextTester
from unittests.text._inputs import _inputs_single_reference

_METRIC_KEY_TO_IDX = {
    "precision": 0,
    "recall": 1,
    "f1": 2,
}

MODEL_NAME = "albert-base-v2"

# Disable tokenizers parallelism (forking not friendly with parallelism)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
def _reference_bert_score(
    preds: Sequence[str],
    target: Sequence[str],
    num_layers: int,
    all_layers: bool,
    idf: bool,
    rescale_with_baseline: bool,
    metric_key: Literal["f1", "precision", "recall"],
) -> Tensor:
    try:
        from bert_score import score as original_bert_score
    except ImportError:
        pytest.skip("test requires bert_score package to be installed.")

    score_tuple = original_bert_score(
        preds,
        target,
        model_type=MODEL_NAME,
        lang="en",
        num_layers=num_layers,
        all_layers=all_layers,
        idf=idf,
        batch_size=len(preds),
        rescale_with_baseline=rescale_with_baseline,
        nthreads=0,
    )
    return score_tuple[_METRIC_KEY_TO_IDX[metric_key]]


@pytest.mark.parametrize(
    ["num_layers", "all_layers", "idf", "rescale_with_baseline", "metric_key"],
    [
        (8, False, False, False, "precision"),
        (12, True, False, False, "recall"),
        (12, False, True, False, "f1"),
        (8, False, False, True, "precision"),
        (12, True, True, False, "recall"),
        (12, True, False, True, "f1"),
        (8, False, True, True, "precision"),
        (12, True, True, True, "f1"),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_single_reference.preds, _inputs_single_reference.target)],
)
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
class TestBERTScore(TextTester):
    """Tests for BERTScore."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @skip_on_connection_issues()
    def test_bertscore_class(self, ddp, preds, targets, num_layers, all_layers, idf, rescale_with_baseline, metric_key):
        """Test the bert score class."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        reference_bert_score_metric = partial(
            _reference_bert_score,
            num_layers=num_layers,
            all_layers=all_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            metric_key=metric_key,
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=BERTScore,
            reference_metric=reference_bert_score_metric,
            metric_args=metric_args,
            key=metric_key,
            check_scriptable=False,  # huggingface transformers are not usually scriptable
            ignore_order=ddp,  # ignore order of predictions when DDP is used
        )

    @skip_on_connection_issues()
    def test_bertscore_functional(self, preds, targets, num_layers, all_layers, idf, rescale_with_baseline, metric_key):
        """Test the bertscore functional."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        reference_bert_score_metric = partial(
            _reference_bert_score,
            num_layers=num_layers,
            all_layers=all_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            metric_key=metric_key,
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=bert_score,
            reference_metric=reference_bert_score_metric,
            metric_args=metric_args,
            key=metric_key,
        )

    def test_bertscore_differentiability(
        self, preds, targets, num_layers, all_layers, idf, rescale_with_baseline, metric_key
    ):
        """Test the bertscore differentiability."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=BERTScore,
            metric_functional=bert_score,
            metric_args=metric_args,
            key=metric_key,
        )


@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.parametrize(
    "idf",
    [(False,), (True,)],
)
def test_bertscore_sorting(idf: bool):
    """Test that BERTScore is invariant to the order of the inputs."""
    short = "Short text"
    long = "This is a longer text"

    preds = [long, long]
    targets = [long, short]

    metric = BERTScore(idf=idf)
    score = metric(preds, targets)

    # First index should be the self-comparison - sorting by length should not shuffle this

@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.parametrize(
    ["idf", "batch_size"],
    [(False, 1),
     (False, 9),
     (True, 1),
     (True, 9)],
)
def test_bertscore_most_similar(idf: bool, batch_size: int):
    """Tests that BERTScore actually gives the highest score to self-similarity."""
    short = "hello there"
    long = "master kenobi"
    longer = "general kenobi"
    
    sentences = [short, long, longer]
    preds, targets = list(zip(*list(product(sentences,
                                            sentences))))
    score = bert_score(preds, targets, idf=idf, lang="en",
                       rescale_with_baseline=False, batch_size=batch_size)
    for i in range(len(preds)):
        max_pred = i%(len(sentences))*(1 + len(sentences))
        max_target = int(i/(len(sentences)))*(1 + len(sentences))
        assert score["f1"][i] <= score["f1"][max_pred], \
            f"pair: {preds[i], targets[i]} does not have a lower score than {preds[max_pred], targets[max_pred]}\n{i=}{max_pred=}"
        assert score["f1"][i] <= score["f1"][max_target], \
            f"pair: {preds[i], targets[i]} does not have a lower score than {preds[max_target], targets[max_target]}\n{i=}{max_target=}"

@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.parametrize(
    ["idf"],
    [(False,),
     (True,)],
)
def test_bertscore_most_similar_separate_calls(idf: bool):
    """Tests that BERTScore actually gives the highest score to self-similarity."""
    short = "hello there"
    long = "master kenobi"
    longer = "general kenobi"
    
    sentences = [short, long, longer]
    pairs_to_compare = product(sentences,
                               sentences)
    preds, targets = list(zip(*list(product(sentences,
                                            sentences))))
    score = {"f1": [bert_score([pred],[target], idf=idf, lang="en",
                                rescale_with_baseline=False)["f1"].item()
                     for pred, target in pairs_to_compare]}
    for i in range(len(preds)):
        max_pred = i%(len(sentences))*(1 + len(sentences))
        max_target = int(i/(len(sentences)))*(1 + len(sentences))
        assert score["f1"][i] <= score["f1"][max_pred], \
            f"pair: {preds[i], targets[i]} does not have a lower score than {preds[max_pred], targets[max_pred]}\n{i=}{max_pred=}"
        assert score["f1"][i] <= score["f1"][max_target], \
            f"pair: {preds[i], targets[i]} does not have a lower score than {preds[max_target], targets[max_target]}\n{i=}{max_target=}"

    
@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.parametrize(
    ["idf", "batch_size"],
    [(False, 1),
     (False, 9),
     (True, 1),
     (True, 9)],
)
def test_bertscore_symmetry(idf: bool, batch_size: int):
    """Tests that BERTscore F1 score is symmetric between reference and prediction.
    As F1 is symmetric, it should also be symmetric."""

    short = "hello there"
    long = "master kenobi"
    longer = "general kenobi"

    sentences = [short, long, longer]
    preds, targets = list(zip(*list(product(sentences,
                                            sentences))))
    score = bert_score(preds, targets, idf=idf, lang="en",
                       rescale_with_baseline=False, batch_size=batch_size)
    for i in range(len(preds)):
        for j in range(len(targets)):
            if preds[i] == targets[j] and preds[j] == targets[i]:
                assert score['f1'][i] == pytest.approx(score['f1'][j]), \
                    f"f1 score for {(preds[i], targets[i])} is not the same as {(preds[j], targets[j])}."
    pass

@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.parametrize(
    ["idf"],
    [(False,),
     (True,)],
)
def test_bertscore_symmetry_separate_calls(idf: bool):
    """Tests that BERTscore F1 score is symmetric between reference and prediction.
    As F1 is symmetric, it should also be symmetric."""
    short = "hello there"
    long = "master kenobi"
    longer = "general kenobi"
    
    sentences = [short, long, longer]
    pairs_to_compare = product(sentences,
                               sentences)
    preds, targets = list(zip(*list(product(sentences,
                                            sentences))))
    score = {"f1": [bert_score([pred],[target], idf=idf, lang="en",
                                rescale_with_baseline=False)["f1"].item()
                     for pred, target in pairs_to_compare]}
    for i in range(len(preds)):
        for j in range(len(targets)):
            if preds[i] == targets[j] and preds[j] == targets[i]:
                assert score['f1'][i] == pytest.approx(score['f1'][j]), \
                    f"f1 score for {(preds[i], targets[i])} is not the same as {(preds[j], targets[j])}."
    pass

@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.parametrize(
    ["idf", "batch_size"],
    [(False, 1),
     (False, 3)]
)
def test_bertscore_additional_sentence(idf: bool, batch_size: int):
    """Tests that BERTscore keeps the same scores for previous inputs
    by adding additional elements to the input lists. This should be the case for idf=False."""

    short = "hello there"
    long = "master kenobi"
    longer = "general kenobi"

    preds = [long,long]
    targets = [long,short]

    score = bert_score(preds, targets, idf=idf, lang="en",
                       rescale_with_baseline=False, batch_size=batch_size)

    longlong = score["f1"][0]
    longshort = score["f1"][1]
    # First index should be the self-comparison - sorting by length should not shuffle this
    assert longlong > longshort
    
    preds = preds + [short, longer]
    targets = targets + [longer, long]

    score = bert_score(preds, targets, idf=idf, lang="en",
                       rescale_with_baseline=False, batch_size=batch_size)

    # First two indices should be exactly as in the previous call to metric
    assert score["f1"][0] == pytest.approx(longlong)
    assert score["f1"][1] == pytest.approx(longshort)
    # Indices 1 and 2 should also be smaller than self-comparison.
    assert score["f1"][0] > score["f1"][1]
    assert score["f1"][0] > score["f1"][2]

