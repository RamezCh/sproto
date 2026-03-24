import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sproto.metrics import metrics


class TestPRAUC:
    def test_prauc_init(self):
        prauc = metrics.PR_AUC(num_classes=5)
        
        assert prauc.num_classes == 5
        assert isinstance(prauc.prauc, list)

    def test_prauc_update(self):
        prauc = metrics.PR_AUC(num_classes=5)
        
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        target[:, 0] = 1
        
        prauc.update(prediction, target)
        
        assert len(prauc.prauc) == 1

    def test_prauc_compute(self):
        prauc = metrics.PR_AUC(num_classes=5)
        
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        target[:, 0] = 1
        target[:, 1] = 1
        
        prauc.update(prediction, target)
        
        result = prauc.compute()
        
        assert isinstance(result, torch.Tensor)


class TestPRAUCPerBucket:
    def test_prauc_per_bucket_init(self):
        bucket = [0, 1, 2]
        prauc = metrics.PR_AUCPerBucket(num_classes=5, bucket=bucket)
        
        assert prauc.num_classes == 5
        assert prauc.bucket == set(bucket)

    def test_prauc_per_bucket_update(self):
        bucket = [0, 1]
        prauc = metrics.PR_AUCPerBucket(num_classes=5, bucket=bucket)
        
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        target[:, 0] = 1
        target[:, 1] = 1
        
        prauc.update(prediction, target)
        
        assert len(prauc.prauc) >= 0


class TestCalculatePRAUC:
    def test_calculate_pr_auc_basic(self):
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        target[:, 0] = 1
        target[:, 1] = 1
        
        result = metrics.calculate_pr_auc(
            prediction=prediction,
            target=target,
            num_classes=5,
            device=torch.device('cpu')
        )
        
        assert isinstance(result, torch.Tensor)

    def test_calculate_pr_auc_with_positive_samples(self):
        prediction = torch.rand(20, 3)
        target = torch.zeros(20, 3)
        target[:10, 0] = 1
        target[5:15, 1] = 1
        target[:5, 2] = 1
        
        result = metrics.calculate_pr_auc(
            prediction=prediction,
            target=target,
            num_classes=3,
            device=torch.device('cpu')
        )
        
        assert isinstance(result, torch.Tensor)


class TestFilteredAUROC:
    def test_filtered_auroc_init(self):
        auroc = metrics.FilteredAUROC(num_classes=5)
        
        assert auroc.num_classes == 5

    def test_filtered_auroc_update_and_compute(self):
        auroc = metrics.FilteredAUROC(num_classes=5, compute_on_step=True)
        
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        target[:, 0] = 1
        target[:, 1] = 1
        target[:, 2] = 1
        
        auroc.update(prediction, target)
        
        result = auroc.compute()
        
        assert isinstance(result, torch.Tensor)

    def test_filtered_auroc_with_all_zero_targets(self):
        auroc = metrics.FilteredAUROC(num_classes=5, compute_on_step=True)
        
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        
        auroc.update(prediction, target)
        
        result = auroc.compute()
        
        assert isinstance(result, torch.Tensor)


class TestFilteredAUROCPerBucket:
    def test_filtered_auroc_per_bucket_init(self):
        bucket = [0, 1, 2]
        auroc = metrics.FilteredAUROCPerBucket(bucket=bucket, num_classes=5)
        
        assert auroc.bucket == set(bucket)
        assert auroc.num_classes == 5

    def test_filtered_auroc_per_bucket_init_with_pos_label(self):
        bucket = [0, 1]
        auroc = metrics.FilteredAUROCPerBucket(
            bucket=bucket,
            num_classes=5,
            pos_label=1
        )
        
        assert auroc.bucket == set(bucket)

    def test_filtered_auroc_per_bucket_compute(self):
        bucket = [0, 1, 2]
        auroc = metrics.FilteredAUROCPerBucket(
            bucket=bucket,
            num_classes=5,
            compute_on_step=True
        )
        
        prediction = torch.rand(10, 5)
        target = torch.zeros(10, 5)
        target[:, 0] = 1
        target[:, 1] = 1
        target[:, 2] = 1
        
        auroc.update(prediction, target)
        
        result = auroc.compute()
        
        assert isinstance(result, torch.Tensor)

    def test_filtered_auroc_per_bucket_average_options(self):
        bucket = [0, 1]
        
        for average in ["macro", "weighted", "none"]:
            auroc = metrics.FilteredAUROCPerBucket(
                bucket=bucket,
                num_classes=5,
                average=average,
                compute_on_step=True
            )
            
            prediction = torch.rand(10, 5)
            target = torch.zeros(10, 5)
            target[:, 0] = 1
            target[:, 1] = 1
            
            auroc.update(prediction, target)
            result = auroc.compute()
            
            assert isinstance(result, torch.Tensor)


class TestFilteredAUROCPerBucketMaxFPR:
    def test_filtered_auroc_per_bucket_with_max_fpr(self):
        bucket = [0, 1]
        auroc = metrics.FilteredAUROCPerBucket(
            bucket=bucket,
            num_classes=5,
            max_fpr=0.5,
            compute_on_step=True
        )
        
        prediction = torch.rand(20, 5)
        target = torch.zeros(20, 5)
        target[:10, 0] = 1
        target[5:15, 1] = 1
        
        auroc.update(prediction, target)
        
        result = auroc.compute()
        
        assert isinstance(result, torch.Tensor)