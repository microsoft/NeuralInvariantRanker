import numpy as np
import torch
from torch import nn
from src.ranker.similatiry_loss import CrossMatchLoss
from src.ranker.util import get_logger

logger = get_logger()


class Codex(nn.Module):
    def __init__(self):
        super().__init__()


class CodexBasedModel(Codex):
    def __init__(
        self,
        hidden_dim: int,
        model_name: str = 'text-ada',
        alpha: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.conversion_layers = nn.ModuleList([
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                bias=True
            ),
            # nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                bias=True
            ),
            # nn.Sigmoid(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
            ),
            # nn.ReLU(),
        ])
        self.drop = nn.Dropout(0.1)
        self.loss_fn = CrossMatchLoss(alpha)

    def get_vector(
        self,
        input_vector: torch.Tensor,  # (B * H)
    ):
        batched = True
        if input_vector.ndim == 1:
            batched = False
            input_vector = input_vector.unsqueeze(0)
        assert input_vector.ndim == 2
        output = input_vector
        for conversion_layer in self.conversion_layers:
            output = conversion_layer(output)
        # output = self.conversion_layer(input_vector)
        if not batched:
            output = output.squeeze(0)
        return output.detach()

    def convert_vector(
        self,
        input_vector: torch.Tensor,  
    ):
        output = input_vector
        for conversion_layer in self.conversion_layers:
            output = conversion_layer(output)
        return output

    def forward(
        self,
        input_vector: torch.Tensor,  # (B * H)
        positive_vectors: torch.Tensor = None,  # (B * P * H)
        negative_vectors: torch.Tensor = None,  # (B * N * H)
        positive_semantic_match_scores: torch.Tensor = None,  # (B * P)
        negative_semantic_match_scores: torch.Tensor = None,  # (B * N)
    ):
        # print(input_vector.shape)
        p = positive_vectors.shape[1]
        n = negative_vectors.shape[1]
        input_vector = self.drop(self.convert_vector(input_vector))
        if p > 0:
            positive_vectors = self.drop(self.convert_vector(positive_vectors))
        if n > 0:
            negative_vectors = self.drop(self.convert_vector(negative_vectors))
        return self.loss_fn(
            input_vector=input_vector,
            positive_vectors=positive_vectors if p > 0 else None,
            negative_vectors=negative_vectors if n > 0 else None,
            positive_semantic_match_scores=positive_semantic_match_scores if p > 0 else None,
            negative_semantic_match_scores=negative_semantic_match_scores if n > 0 else None,
        )


class CodexBasedClassificationModel(Codex):
    def __init__(
        self,
        hidden_dim: int,
        model_name: str = 'text-ada',
        use_binary: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.conversion_layers = nn.ModuleList([
            nn.Linear(
                in_features=2*self.hidden_dim,
                out_features=self.hidden_dim,
                bias=True
            ),
            # nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                bias=True
            ),
            # nn.Sigmoid(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
            ),
            # nn.ReLU(),
        ])
        if use_binary:
            self.num_classes = 2
        else:
            self.num_classes = 5
        self.drop = nn.Dropout(0.1)
        self.classification_layer = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.num_classes,
        )
        self.activation = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_vector(
        self,
        input_vector: torch.Tensor,  # (B * H)
    ):
        raise NotImplementedError

    def convert_vector(
        self,
        input_vector: torch.Tensor,  
    ):
        output = input_vector
        for conversion_layer in self.conversion_layers:
            output = conversion_layer(output)
        return output

    def get_scores(
        self,
        problem_vector: torch.Tensor,  # (B * H),
        invariant_vectors: torch.Tensor,  # (B * D * H)
    ):
        if isinstance(problem_vector, np.ndarray):
            problem_vector = torch.FloatTensor(
                problem_vector, 
                device=self.classification_layer.weight.device
            )
        if isinstance(invariant_vectors, np.ndarray):
            invariant_vectors = torch.FloatTensor(
                invariant_vectors, 
                device=self.classification_layer.weight.device
            )
        # logger.info(f"problem_vector.shape: {problem_vector.shape}")
        # logger.info(f"invariant_vectors.shape: {invariant_vectors.shape}")
        batched = True
        if problem_vector.ndim == 1:
            problem_vector = problem_vector.unsqueeze(0)
            batched = False
        if invariant_vectors.ndim == 2:
            assert not batched
            invariant_vectors = invariant_vectors.unsqueeze(0)
        assert problem_vector.ndim == 2 and invariant_vectors.ndim == 3
        problem_repeated = problem_vector.unsqueeze(1).repeat(1, invariant_vectors.shape[1], 1)

        input_vector = torch.cat([problem_repeated, invariant_vectors], dim=-1)
        output_vector = self.drop(self.convert_vector(input_vector))
        scores = self.activation(self.classification_layer(output_vector))
        full_scores = scores[:, :, -1].detach().cpu()
        if not batched:
            full_scores = full_scores.squeeze(0)
        return full_scores.numpy()

    def forward(
        self,
        input_vector: torch.Tensor,  # (B * H)
        positive_vectors: torch.Tensor = None,  # (B * P * H)
        negative_vectors: torch.Tensor = None,  # (B * N * H)
        positive_semantic_match_scores: torch.Tensor = None,  # (B * P)
        negative_semantic_match_scores: torch.Tensor = None,  # (B * N)
    ):
        # print(input_vector.shape)
        b, h = input_vector.shape
        if positive_vectors is None:
            positive_vectors = torch.zeros((b, 0, h)).to(input_vector.device)
        if negative_vectors is None:
            negative_vectors = torch.zeros((b, 0, h)).to(input_vector.device)
        p = positive_vectors.shape[1]
        n = negative_vectors.shape[1]
        assert p + n > 0, "At least one positive or negative vector must be provided"
        if self.num_classes == 2:
            positive_labels = torch.ones((input_vector.shape[0], p), dtype=torch.long
            ).to(input_vector.device)
            negative_labels = torch.zeros((input_vector.shape[0], n), dtype=torch.long
            ).to(input_vector.device)
        else:
            assert (
                (p == 0 or positive_semantic_match_scores is not None) and 
                (n == 0 or negative_semantic_match_scores is not None)
            ), "Semantic match scores must be provided for multi-class classification"
            if p > 0:
                positive_labels = (
                    positive_semantic_match_scores * 4
                ).long().to(input_vector.device)
            else:
                positive_labels = (
                    torch.zeros((input_vector.shape[0], p), dtype=torch.long)
                ).to(input_vector.device)
            if n > 0:
                negative_labels = (
                    negative_semantic_match_scores * 4
                ).long().to(input_vector.device)
            else:
                negative_labels = (
                    torch.zeros((input_vector.shape[0], n), dtype=torch.long)
                ).to(input_vector.device)
            
        complete_labels = torch.cat(
            (positive_labels, negative_labels), 
            dim=1
        )
        input_repeat = input_vector.unsqueeze(1).repeat(1, p+n, 1)
        target_vectors = torch.cat([positive_vectors, negative_vectors], dim=1)
        input_target_vectors = torch.cat([input_repeat, target_vectors], dim=-1)
        input_target_vectors = self.drop(self.convert_vector(input_target_vectors))
        logits = self.classification_layer(input_target_vectors)
        reshaped_logits = logits.reshape(-1, self.num_classes)
        reshaped_labels = complete_labels.reshape(-1)
        loss = self.loss_fn(reshaped_logits, reshaped_labels)
        positive_logits = self.activation(logits[:, :p, :])[:, :, -1]
        negative_logits = self.activation(logits[:, p:, :])[:, :, -1]
        return {
            'loss': loss,
            'scores': {
                'positive': positive_logits,
                'negative': negative_logits,
            },
        }
        

if __name__ == '__main__':
    model = CodexBasedClassificationModel(
        hidden_dim=16, 
        use_binary=False
    )
    input_vector = torch.randn(2, 16)
    positive_vectors = torch.randn(2, 6, 16)
    negative_vectors = torch.randn(2, 1, 16)
    positve_semantic = np.random.choice([0, .25, .5, .75, 1], size=(2, 6))
    negative_semantic = np.random.choice([0, .25, .5, .75, 1], size=(2, 1))
    positve_semantic = torch.tensor(positve_semantic, dtype=torch.float32)
    negative_semantic = torch.tensor(negative_semantic, dtype=torch.float32)
    ret = model(
        input_vector=input_vector,
        positive_vectors=None,
        negative_vectors=negative_vectors,
        positive_semantic_match_scores=positve_semantic,
        negative_semantic_match_scores=negative_semantic,
    )
    print(ret['loss'])
    print(ret['scores']['positive'].shape)
    print(ret['scores']['negative'].shape)
    scores = model.get_scores(
        problem_vector=input_vector[0].detach().cpu().numpy(),
        invariant_vectors=positive_vectors[0].cpu().numpy(),
    )
    print(scores)

