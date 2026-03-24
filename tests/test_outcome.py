import pytest
import torch
import numpy as np
import pandas as pd
import pickle
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sproto.dataset import outcome


class TestCollateBatch:
    def test_collate_batch_basic(self):
        featurized_samples = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "tokens": ["hello", "world", "test"],
                "target": [1, 0, 1],
                "sample_id": "123"
            },
            {
                "input_ids": [4, 5, 6, 7],
                "attention_mask": [1, 1, 1, 1],
                "tokens": ["foo", "bar", "baz", "qux"],
                "target": [0, 1, 0],
                "sample_id": "456"
            }
        ]
        
        result = outcome.collate_batch(featurized_samples)
        
        assert "input_ids" in result
        assert "attention_masks" in result
        assert "tokens" in result
        assert "targets" in result
        assert "sample_ids" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_masks"], torch.Tensor)

    def test_collate_batch_with_token_type_ids(self):
        featurized_samples = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
                "tokens": ["hello", "world", "test"],
                "target": [1, 0],
                "sample_id": "123"
            },
            {
                "input_ids": [4, 5, 6],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
                "tokens": ["foo", "bar", "baz"],
                "target": [0, 1],
                "sample_id": "456"
            }
        ]
        
        result = outcome.collate_batch(featurized_samples)
        
        assert "token_type_ids" in result

    def test_collate_batch_single_sample(self):
        featurized_samples = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "tokens": ["hello", "world", "test"],
                "target": [1, 0, 1],
                "sample_id": "123"
            }
        ]
        
        result = outcome.collate_batch(featurized_samples)
        
        assert result["input_ids"].shape[0] == 1
        assert result["attention_masks"].shape[0] == 1


class TestSampleToFeaturesMultilabel:
    def test_sample_to_features_multilabel_basic(self):
        class MockTokenizer:
            def encode_plus(self, text, padding, truncation, pad_to_multiple_of, max_length):
                return {
                    "input_ids": [1, 2, 3, 4, 0],
                    "attention_mask": [1, 1, 1, 1, 0],
                    "token_type_ids": [0, 0, 0, 0, 0]
                }
            
            @property
            def encodings(self):
                class MockEncoding:
                    tokens = ["[CLS]", "hello", "world", "[SEP]", "[PAD]"]
                return [MockEncoding()]
        
        sample = pd.Series({
            "text": "hello world",
            "label1": 1,
            "label2": 0,
            "label3": 1,
            "hadm_id": "12345"
        })
        
        tokenizer = MockTokenizer()
        labels = ["label1", "label2", "label3"]
        
        result = outcome.sample_to_features_multilabel(
            sample=sample,
            tokenizer=tokenizer,
            labels=labels
        )
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "tokens" in result
        assert "target" in result
        assert "sample_id" in result
        assert result["sample_id"] == "12345"
        assert len(result["target"]) == 3

    def test_sample_to_features_multilabel_without_token_type_ids(self):
        class MockTokenizer:
            def encode_plus(self, text, padding, truncation, pad_to_multiple_of, max_length):
                return {
                    "input_ids": [1, 2, 3, 4],
                    "attention_mask": [1, 1, 1, 1]
                }
            
            @property
            def encodings(self):
                class MockEncoding:
                    tokens = ["[CLS]", "hello", "world", "[SEP]"]
                return [MockEncoding()]
        
        sample = pd.Series({
            "text": "test text",
            "label1": 1,
            "label2": 0,
            "hadm_id": "999"
        })
        
        tokenizer = MockTokenizer()
        labels = ["label1", "label2"]
        
        result = outcome.sample_to_features_multilabel(
            sample=sample,
            tokenizer=tokenizer,
            labels=labels
        )
        
        assert "token_type_ids" not in result


class TestOutcomeDiagnosesDataset:
    def test_dataset_init_with_csv(self):
        class MockTokenizer:
            def encode_plus(self, text, padding, truncation, pad_to_multiple_of, max_length):
                return {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "token_type_ids": [0, 0, 0]
                }
            
            @property
            def encodings(self):
                class MockEncoding:
                    tokens = ["[CLS]", "test", "[SEP]"]
                return [MockEncoding()]
        
        test_data = """hadm_id,text,main_ccsr_code_encoded
12345,"test text","['code1', 'code2']"
67890,"another text","['code1']"
"""
        
        all_codes = ["code1", "code2", "code3"]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write(test_data)
            csv_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(all_codes, f)
            codes_path = f.name
        
        try:
            dataset = outcome.OutcomeDiagnosesDataset(
                file_path=csv_path,
                tokenizer=MockTokenizer(),
                all_codes_path=codes_path
            )
            
            assert len(dataset) == 2
        finally:
            os.unlink(csv_path)
            os.unlink(codes_path)

    def test_dataset_init_with_dataframe(self):
        class MockTokenizer:
            def encode_plus(self, text, padding, truncation, pad_to_multiple_of, max_length):
                return {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1]
                }
            
            @property
            def encodings(self):
                class MockEncoding:
                    tokens = ["[CLS]", "test", "[SEP]"]
                return [MockEncoding()]
        
        df = pd.DataFrame({
            "hadm_id": ["123", "456"],
            "text": ["text1", "text2"],
            "main_ccsr_code_encoded": ["['c1']", "['c2']"]
        })
        
        all_codes = ["c1", "c2"]
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(all_codes, f)
            codes_path = f.name
        
        try:
            dataset = outcome.OutcomeDiagnosesDataset(
                file_path="dummy.csv",
                tokenizer=MockTokenizer(),
                all_codes_path=codes_path,
                data=df
            )
            
            assert len(dataset) == 2
        finally:
            os.unlink(codes_path)

    def test_dataset_getitem(self):
        class MockTokenizer:
            def encode_plus(self, text, padding, truncation, pad_to_multiple_of, max_length):
                return {
                    "input_ids": [1, 2, 3, 0],
                    "attention_mask": [1, 1, 1, 0],
                    "token_type_ids": [0, 0, 0, 0]
                }
            
            @property
            def encodings(self):
                class MockEncoding:
                    tokens = ["[CLS]", "test", "[SEP]", "[PAD]"]
                return [MockEncoding()]
        
        df = pd.DataFrame({
            "hadm_id": ["123"],
            "text": ["test text"],
            "main_ccsr_code_encoded": ["['c1']"]
        })
        
        all_codes = ["c1", "c2"]
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(all_codes, f)
            codes_path = f.name
        
        try:
            dataset = outcome.OutcomeDiagnosesDataset(
                file_path="dummy.csv",
                tokenizer=MockTokenizer(),
                all_codes_path=codes_path,
                data=df
            )
            
            item = dataset[0]
            
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "tokens" in item
            assert "target" in item
            assert "sample_id" in item
        finally:
            os.unlink(codes_path)

    def test_dataset_get_num_classes(self):
        class MockTokenizer:
            def encode_plus(self, text, padding, truncation, pad_to_multiple_of, max_length):
                return {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1]
                }
            
            @property
            def encodings(self):
                class MockEncoding:
                    tokens = ["[CLS]", "test", "[SEP]"]
                return [MockEncoding()]
        
        df = pd.DataFrame({
            "hadm_id": ["123"],
            "text": ["test text"],
            "main_ccsr_code_encoded": ["['c1']"]
        })
        
        all_codes = ["c1", "c2", "c3"]
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(all_codes, f)
            codes_path = f.name
        
        try:
            dataset = outcome.OutcomeDiagnosesDataset(
                file_path="dummy.csv",
                tokenizer=MockTokenizer(),
                all_codes_path=codes_path,
                data=df
            )
            
            num_classes = dataset.get_num_classes()
            
            assert num_classes == 3
        finally:
            os.unlink(codes_path)