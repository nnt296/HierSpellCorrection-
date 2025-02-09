import torch
from torch import nn
from torch.nn import functional as F
# from fvcore.nn.focal_loss import sigmoid_focal_loss


def compute_detection_loss(
        detection_logits: torch.Tensor,
        detection_labels: torch.Tensor
):
    """
    Compute detection loss

    Notes:
    This function differs in terms of sequence length normalization.
    The L_detection does not account loss for special tokens, but ours does.

    Args:
        detection_logits: output of detection classifier of shape B x seq_len x 2
        detection_labels: binary label of shape B x seq_len
                         0 for correct and 1 for incorrect
                         -100 to ignore
        loss
    """
    criteria = nn.CrossEntropyLoss(reduction="none")

    _, max_len, _ = detection_logits.size()

    _det_logits = detection_logits.view(-1, 2)
    _det_labels = detection_labels.view(-1)

    valid_indexes = torch.where(_det_labels != -100)[0]
    assert len(valid_indexes) > 0, "[WARNING] Empty sentence!"

    # Temporary fix the hyperparameter here
    # loss = sigmoid_focal_loss(_det_logits, _det_labels, alpha=-1, gamma=2, reduction="mean")

    loss: torch.Tensor = criteria(_det_logits, _det_labels)  # B x max_len
    
    # Normalize the loss based on length of the sequence
    # (Follow the paper but not sure if this has any effect)
    _det_labels = detection_labels != -100
    seq_len = _det_labels.sum(dim=1, keepdim=False).type(loss.dtype)
    loss = loss.view(-1, max_len)
    loss = loss * _det_labels  # Mask out pad tokens
    loss = loss.sum(dim=1, keepdim=False)
    loss = torch.sum(loss / (seq_len + 1e-5))
    return loss


def compute_correct_loss(
        correction_logits: torch.Tensor,
        detection_labels: torch.Tensor,
        correction_labels: torch.Tensor
):
    """
    Compute correction loss only on truly error tokens

    Notes:
    This loss function MIGHT NOT MATCH the paper's L_correction
    Tokens with indices set to `-100` are ignored
    Loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

    Args:
        correction_logits: output of detection classifier of shape B x seq_len x num_vocab
        detection_labels: binary label of shape B x seq_len
        correction_labels: correct label of miss-spelled tokens of shape B x seq_len
    Returns:
        loss
    """
    criteria = nn.CrossEntropyLoss(reduction="none")
    
    _, max_len, num_classes = correction_logits.size()
    
    _corr_logits = correction_logits.view(-1, num_classes)
    _corr_labels = correction_labels.view(-1)

    loss: torch.Tensor = criteria(_corr_logits, _corr_labels)  # B x max_len

    # Normalize the loss based on length of the sequence
    _det_labels = detection_labels > 0
    seq_len = _det_labels.sum(dim=1, keepdim=False).type(loss.dtype)
    loss = loss.view(-1, max_len)
    loss = loss * _det_labels  # Mask out pad tokens
    loss = loss.sum(dim=1, keepdim=False)
    loss = torch.sum(loss / (seq_len + 1e-5))

    return loss


if __name__ == '__main__':
    torch.manual_seed(42)

    num_vocab = 100

    d_logits = torch.randn(2, 10, 2)  # Shape B x seq_len x 2
    d_labels = torch.randint(low=0, high=2, size=(2, 10))  # Shap B x seq_len
    d_labels[1][-2:] = -100

    c_logits = torch.randn(2, 10, num_vocab)  # Shape B x seq_len x num_vocabs
    c_labels = torch.randint(low=0, high=num_vocab, size=(2, 10))  # Shap B x seq_len

    print(d_labels)
    print(c_labels)

    d_loss = compute_detection_loss(d_logits, d_labels)
    print(d_loss)

    c_loss = compute_correct_loss(c_logits, d_labels, c_labels)
    print(c_loss)
