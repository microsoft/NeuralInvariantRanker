# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
import numpy as np


class CrossMatchLoss(nn.Module):
    def __init__(
        self,
        alpha=0.,
    ):
        super().__init__()
        self.alpha = alpha
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(
        self,
        input_vector: torch.Tensor,  # (B * H)
        positive_vectors: torch.Tensor = None,  # (B * P * H)
        negative_vectors: torch.Tensor = None,  # (B * N * H)
        positive_semantic_match_scores: torch.Tensor = None,  # (B * P)
        negative_semantic_match_scores: torch.Tensor = None,  # (B * N)
    ):
        if positive_vectors is None and negative_vectors is None:
            raise ValueError(
                "CrossMatchLoss does not know how to calculate the loss if" +
                "Both the positive vectors and negative vectors are None." +
                "Please provide at least one non-None vectors"
            )
        if positive_semantic_match_scores is None \
                and negative_semantic_match_scores is None:
            alpha = 0.
        else:
            alpha = self.alpha

        input_norm = torch.norm(
            input_vector, dim=-1, keepdim=True, p=2
        ).unsqueeze(1)
        modified_input_vector = input_vector.unsqueeze(1)
        if positive_vectors is not None:
            positive_norm = torch.norm(
                positive_vectors, dim=-1, keepdim=True, p=2
            )
            positive_products = torch.matmul(
                input_norm, positive_norm.transpose(1, 2)
            ).squeeze(1)
            modified_pv = positive_vectors.transpose(1, 2)
            positive_scores = torch.abs(
                torch.matmul(modified_input_vector, modified_pv)
            ).squeeze(1)
            positive_scores = positive_scores / positive_products
            positive_labels = torch.ones_like(positive_scores)
        else:
            positive_scores = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
            positive_labels = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
        if positive_vectors is None or positive_semantic_match_scores is None:
            positive_semantic_match_scores = torch.zeros_like(positive_scores)

        if negative_vectors is not None:
            negative_norm = torch.norm(
                negative_vectors, dim=-1, keepdim=True, p=2
            )
            negative_products = torch.matmul(
                input_norm, negative_norm.transpose(1, 2)
            ).squeeze(1)
            modified_nv = negative_vectors.transpose(1, 2)
            negative_scores = torch.abs(
                torch.matmul(modified_input_vector, modified_nv)
            ).squeeze(1)
            negative_scores = negative_scores / negative_products
            negative_labels = torch.zeros_like(negative_scores)
        else:
            negative_scores = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
            negative_labels = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
        if negative_vectors is None or negative_semantic_match_scores is None:
            negative_semantic_match_scores = torch.zeros_like(negative_scores)

        labels = torch.cat([positive_labels, negative_labels], dim=-1)
        scores = torch.cat([positive_scores, negative_scores], dim=-1)
        semantic_match_scores = torch.cat(
            [positive_semantic_match_scores, negative_semantic_match_scores],
            dim=-1
        )
        labels = alpha * semantic_match_scores + \
            (1 - alpha) * labels
        loss = self.loss_fn(scores, labels)
        return {
            "loss": loss,
            "scores": {
                "positive": positive_scores,
                "negative": negative_scores
            },
            "input_vector": input_vector,
            "positive_vectors": positive_vectors,
            "negative_vectors": negative_vectors,
            "semantic_match_factor": alpha,
        }


if __name__ == '__main__':
    B, H, P, N = 32, 768, 7, 5
    input_vector = torch.FloatTensor(np.random.normal(100, 70, size=(B, H)))
    positive_vectors = torch.FloatTensor(
        np.random.normal(100, 70, size=(B, P, H)))
    negative_vectors = torch.FloatTensor(
        np.random.normal(100, 70, size=(B, N, H)))
    positive_scores = torch.FloatTensor(np.random.uniform(0.8, 1, size=(B, P)))
    negative_scores = torch.FloatTensor(np.random.uniform(0, 0.2, size=(B, N)))
    model = CrossMatchLoss(alpha=.1)
    final_loss = model(
        input_vector=input_vector,
        positive_vectors=positive_vectors,
        negative_vectors=negative_vectors,
        positive_semantic_match_scores=positive_scores,
        negative_semantic_match_scores=None
    )
    print(final_loss["loss"])
    print(final_loss["scores"]["positive"].shape)
    print(final_loss["scores"]["negative"].shape)
    print(final_loss["semantic_match_factor"])
