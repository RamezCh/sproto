import pytest
import torch
import numpy as np
import json
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sproto.utils import utils


class TestFreezeModelWeights:
    def test_freeze_model_weights(self):
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            assert param.requires_grad is True
        
        utils.freeze_model_weights(model)
        
        for param in model.parameters():
            assert param.requires_grad is False


class TestAttentionMaskFromTokens:
    def test_attention_mask_from_tokens_basic(self):
        masks = torch.ones(1, 20)
        token_list = [
            ["hello", "world", "chief", "complaint", ":", "fever"]
        ]
        
        result = utils.attention_mask_from_tokens(masks.clone(), token_list)
        
        assert result[0, 2].item() == 0
        assert result[0, 3].item() == 0
        assert result[0, 4].item() == 0

    def test_attention_mask_from_tokens_cls_sep(self):
        masks = torch.ones(1, 10)
        token_list = [
            ["[CLS]", "hello", "world", "[SEP]"]
        ]
        
        result = utils.attention_mask_from_tokens(masks.clone(), token_list)
        
        assert result[0, 0].item() == 0
        assert result[0, 3].item() == 0

    def test_attention_mask_from_tokens_multiple_patterns(self):
        masks = torch.ones(2, 30)
        token_list = [
            ["chief", "complaint", ":", "fever", "medical", "history", ":", "diabetes"],
            ["physical", "exam", ":", "normal", "family", "history", ":", "hypertension"]
        ]
        
        result = utils.attention_mask_from_tokens(masks.clone(), token_list)
        
        assert result[0, 0].item() == 0
        assert result[0, 1].item() == 0
        assert result[0, 2].item() == 0
        assert result[1, 0].item() == 0
        assert result[1, 1].item() == 0
        assert result[1, 2].item() == 0


class TestPadBatchSamples:
    def test_pad_batch_samples_no_padding_needed(self):
        batch_samples = [["a", "b", "c"]]
        result = utils.pad_batch_samples(batch_samples, 3)
        assert result == ["a", "b", "c"]

    def test_pad_batch_samples_with_padding(self):
        batch_samples = [["a", "b"]]
        result = utils.pad_batch_samples(batch_samples, 4)
        assert result == ["a", "b", "[PAD]", "[PAD]"]

    def test_pad_batch_samples_multiple_samples(self):
        batch_samples = [["a", "b"], ["c", "d", "e"]]
        result = utils.pad_batch_samples(batch_samples, 3)
        assert result == ["a", "b", "[PAD]", "c", "d", "e"]

    def test_pad_batch_samples_empty(self):
        batch_samples = [[]]
        result = utils.pad_batch_samples(batch_samples, 3)
        assert result == ["[PAD]", "[PAD]", "[PAD]"]


class TestLoadEvalBuckets:
    def test_load_eval_buckets_none_path(self):
        result = utils.load_eval_buckets(None)
        assert result is None

    def test_load_eval_buckets_valid_file(self):
        test_data = {"bucket1": [1, 2, 3], "bucket2": [4, 5]}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = utils.load_eval_buckets(temp_path)
            assert result == test_data
        finally:
            os.unlink(temp_path)

    def test_load_eval_buckets_invalid_path(self):
        result = utils.load_eval_buckets("/nonexistent/path/buckets.json")
        assert result is None


class TestBuildHeatmaps:
    def test_build_heatmaps_basic(self):
        case_tokens = ["hello", "world", "test"]
        token_scores = [[0.1, 0.5, 0.8]]
        
        result = utils.build_heatmaps(case_tokens, token_scores)
        
        assert len(result) == 1
        assert "hello" in result[0]
        assert "world" in result[0]
        assert "test" in result[0]

    def test_build_heatmaps_red_tint(self):
        case_tokens = ["fever"]
        token_scores = [[0.5]]
        
        result = utils.build_heatmaps(case_tokens, token_scores, tint="red")
        
        assert len(result) == 1

    def test_build_heatmaps_blue_tint(self):
        case_tokens = ["fever"]
        token_scores = [[0.5]]
        
        result = utils.build_heatmaps(case_tokens, token_scores, tint="blue")
        
        assert len(result) == 1

    def test_build_heatmaps_multiple_prototypes(self):
        case_tokens = ["hello", "world"]
        token_scores = [[0.1, 0.2], [0.3, 0.4]]
        
        result = utils.build_heatmaps(case_tokens, token_scores)
        
        assert len(result) == 2

    def test_build_heatmaps_with_subtokens(self):
        case_tokens = ["hello", "##world", "test"]
        token_scores = [[0.1, 0.5, 0.3]]
        
        result = utils.build_heatmaps(case_tokens, token_scores)
        
        assert "world" in result[0]


class TestRemoveDir:
    def test_remove_dir_existing(self):
        test_dir = tempfile.mkdtemp()
        
        utils.remove_dir(test_dir)
        
        assert not os.path.exists(test_dir)

    def test_remove_dir_nonexistent(self):
        utils.remove_dir("/nonexistent/directory/path")