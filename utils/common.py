import re


# Word and Char special tokens share the same names, but different in nature
class SpecialTokens:
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    cls: str = "[CLS]"
    sep: str = "[SEP]"


all_special_tokens = [SpecialTokens.pad, SpecialTokens.unk, SpecialTokens.cls, SpecialTokens.sep]


def de_emojify(text):
    regrex_pattern = re.compile(
        pattern="["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "]+", flags=re.UNICODE)
    #     return regrex_pattern.findall(text)
    return regrex_pattern.sub(r'', text)
