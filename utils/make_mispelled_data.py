"""
This script is used to make diacritic errors, ngọng
"""

import random
import numpy as np
import string
import regex

# Constants
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ' \
     u'ẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAa' \
     u'EeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

s3 = u'ẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẾếỀềỂểỄễỆệỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỨứỪừỬửỮữỰự'
s2 = u'ÂâÂâÂâÂâÂâĂăĂăĂăĂăĂăÊêÊêÊêÊêÊêÔôÔôÔôÔôÔôƠơƠơƠơƠơƠơƯưƯưƯưƯưƯư'
alphabet = u'abcdefghijklmnopqrstuvwxyz'

s5 = ['úy', 'ùy', 'ủy', 'ũy', 'ụy', 'óa', 'òa', 'ỏa', 'õa', 'ọa']
s4 = ['uý', 'uỳ', 'uỷ', 'uỹ', 'uỵ', 'oá', 'oà', 'oả', 'oã', 'oạ']

vowels = ['a', 'á', 'à', 'ả', 'ạ', 'ă', 'ắ', 'ằ', 'ẳ', 'ặ', 'â', 'ấ', 'ầ', 'ẩ', 'ậ',
          'e', 'é', 'è', 'ẻ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ệ', 'i', 'í', 'ì', 'ỉ', 'ị',
          'o', 'ó', 'ò', 'ỏ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ợ',
          'u', 'ú', 'ù', 'ủ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ự', 'y', 'ý', 'ỳ', 'ỷ', 'ỵ']

non_vocab_pattern = regex.compile(r"""[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ]""")
# non_vocab_pattern = regex.compile(r"""[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ
# fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳ
# ỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~[:space:]]""")

lisp_dict = {
    "l": ["n"], "n": ["l"],
    "s": ["x"], "x": ["s"],
    "tr": ["ch"], "ch": ["tr"],
    "gi": ["d", "r"], "d": ["gi", "r"], "r": ["gi", "d"]
}


def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s


def manual_replace(s, char, index, length):
    return s[:index] + char + s[index + length:]


def has_numbers_or_non_latin(text):
    return bool(non_vocab_pattern.search(text))


def remove_accent(token):
    if random.random() < 0.5:
        token = remove_accents(token)
    else:
        new_chars = []
        for cc in token:
            if cc in s3 and random.random() < 0.7:
                cc = s2[s3.index(cc)]
            if cc in s1 and random.random() < 0.5:
                cc = s0[s1.index(cc)]
            new_chars.append(cc)
        token = "".join(new_chars)
    return token


def gen_lisp(token):
    for vowel in lisp_dict:
        if token.lower().startswith(vowel):
            alter = random.choice(lisp_dict[vowel])
            if random.random() < 0.5:
                alter = alter.upper()
            token = manual_replace(token, alter, 0, len(vowel))
    return token


def swap_char(token):
    chars = list(token)
    n_swap = min(len(chars), np.random.poisson(0.5) + 1)
    index = np.random.choice(
        np.arange(len(chars)), size=n_swap, replace=False)
    swap_index = index[np.random.permutation(index.shape[0])]
    swap_dict = {ii: jj for ii, jj in zip(index, swap_index)}
    chars = [chars[ii] if ii not in index else chars[swap_dict[ii]]
             for ii in range(len(chars))]
    token = "".join(chars)
    return token


def remove_chars(token):
    n_remove = min(len(token), np.random.poisson(0.005) + 1)
    for _ in range(n_remove):
        pos = np.random.choice(np.arange(len(token)), size=1)[0]
        token = token[:pos] + token[pos + 1:]
    return token


def add_chars(token):
    n_add = min(len(token), np.random.poisson(0.05) + 1)
    adding_chars = np.random.choice(
        list(alphabet), size=n_add, replace=True)
    for cc in adding_chars:
        pos = np.random.choice(np.arange(len(token)), size=1)[0]
        token = "".join([token[:pos], cc, token[pos:]])
    return token


# original typo generation function - not use anymore
def generate_typos(token,
                   typo_prob=0.3,  # Result in 15% error
                   accent_prob=0.5,
                   lisp_prob=0.5,
                   # lowercase_prob=0.5,
                   swap_char_prob=0.2,
                   add_chars_prob=0.2,
                   remove_chars_prob=0.2):
    # Check if successfully altered the token
    origin = token

    # Skip if the token contains number or non-latin characters
    # Including time or measurement: 10h30, 100mg, etc.
    # This typo does not guarantee exact prob
    if random.random() > typo_prob \
            or has_numbers_or_non_latin(text=token) \
            or token in string.punctuation:
        return origin, 0

    opt_probabilities = [5, 5, 2, 5, 5, 5]
    choices = [0, 1, 2, 3, 4, 5]
    opt = random.choices(choices, opt_probabilities)[0]

    if opt == 0:
        # Mix
        if random.random() < accent_prob:
            token = remove_accent(token)
        # Ngong l-n, tr-ch, ...
        if random.random() < lisp_prob:
            token = gen_lisp(token)
        # if random.random() < lowercase_prob:
        #     token = token.lower()
        if random.random() < swap_char_prob:
            token = swap_char(token)
        if random.random() < remove_chars_prob:
            token = remove_chars(token)
        if random.random() < add_chars_prob:
            token = add_chars(token)
    elif opt == 1:
        token = remove_accent(token)
    elif opt == 2:
        token = gen_lisp(token)
    elif opt == 3:
        token = swap_char(token)
    elif opt == 4:
        token = remove_chars(token)
    else:
        token = add_chars(token)

    if len(token) == 0 or token == origin:
        return origin, 0
    else:
        return token, 1
