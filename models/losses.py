import torch
from torch import nn


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
    Returns:
        loss
    """
    criteria = nn.CrossEntropyLoss()

    loss = criteria(detection_logits.view(-1, 2), detection_labels.view(-1))
    # Normalize the loss based on length of the sequence
    # (Follow the paper but not sure if this has any effect)
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
    criteria = nn.CrossEntropyLoss()
    num_classes = correction_logits.size(2)

    _corr_logits = correction_logits.view(-1, num_classes)
    _det_labels = detection_labels.view(-1)
    _corr_labels = correction_labels.view(-1)

    valid_indexes = torch.nonzero(_det_labels, as_tuple=True)[0]

    # Case correct batches, return 0
    if valid_indexes.size(0) == 0:
        return 0

    _corr_logits = torch.index_select(_corr_logits, 0, valid_indexes)
    _corr_labels = torch.index_select(_corr_labels, 0, valid_indexes)

    loss = criteria(_corr_logits, _corr_labels)
    return loss


if __name__ == '__main__':
    torch.manual_seed(22)

    num_vocab = 100

    d_logits = torch.randn(2, 10, 2)  # Shape B x seq_len x 2
    sp_labels = torch.randint(low=0, high=2, size=(2, 10))  # Shap B x seq_len

    c_logits = torch.randn(2, 10, num_vocab)  # Shape B x seq_len x num_vocabs
    tk_labels = torch.randint(low=0, high=num_vocab, size=(2, 10))  # Shap B x seq_len

    print(sp_labels)
    print(tk_labels)

    d_loss = compute_detection_loss(d_logits, sp_labels)
    print(d_loss)

    c_loss = compute_correct_loss(c_logits, sp_labels, tk_labels)
    print(c_loss)
