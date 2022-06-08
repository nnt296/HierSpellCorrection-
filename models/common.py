# Word and Char special tokens share the same names, but different in nature
class SpecialTokens:
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    cls: str = "[CLS]"
    sep: str = "[SEP]"


all_special_tokens = [SpecialTokens.pad, SpecialTokens.unk, SpecialTokens.cls, SpecialTokens.sep]
