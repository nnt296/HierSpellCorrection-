import numpy as np
import re
import unidecode
import nltk
from nltk.tokenize import word_tokenize
import string


# nltk.download('punkt')


class SynthesizeData(object):
    """
    Utils class to create artificial miss-spelled words
    Args:
        vocab_path: path to vocab file.
                    Vocab file is expected to be a set of words, separate by ' ', no newline character.
    """

    def __init__(self, vocab_path=""):
        # self.vocab = open(vocab_path, 'r', encoding = 'utf-8').read().split()
        self.tokenizer = word_tokenize
        self.word_couples = [
            ['sương', 'xương'], ['sĩ', 'sỹ'], ['sẽ', 'sẻ'], ['sã', 'sả'], ['sả', 'xả'], ['sẽ', 'sẻ'],
            ['mùi', 'muồi'],
            ['chỉnh', 'chỉn'], ['sữa', 'sửa'], ['chuẩn', 'chẩn'], ['lẻ', 'lẽ'], ['chẳng', 'chẵng'],
            ['cổ', 'cỗ'],
            ['sát', 'xát'], ['cập', 'cặp'], ['truyện', 'chuyện'], ['xá', 'sá'], ['giả', 'dả'],
            ['đỡ', 'đở'],
            ['giữ', 'dữ'], ['giã', 'dã'], ['xảo', 'sảo'], ['kiểm', 'kiễm'], ['cuộc', 'cục'],
            ['dạng', 'dạn'],
            ['tản', 'tảng'], ['ngành', 'nghành'], ['nghề', 'ngề'], ['nổ', 'nỗ'], ['rảnh', 'rãnh'],
            ['sẵn', 'sẳn'],
            ['sáng', 'xán'], ['xuất', 'suất'], ['suôn', 'suông'], ['sử', 'xử'], ['sắc', 'xắc'],
            ['chữa', 'chửa'],
            ['thắn', 'thắng'], ['dỡ', 'dở'], ['trải', 'trãi'], ['trao', 'trau'], ['trung', 'chung'],
            ['thăm', 'tham'],
            ['sét', 'xét'], ['dục', 'giục'], ['tả', 'tã'], ['sông', 'xông'], ['sáo', 'xáo'],
            ['sang', 'xang'],
            ['ngã', 'ngả'], ['xuống', 'suống'], ['xuồng', 'suồng']
        ]

        self.vn_alphabet = [
            'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô',
            'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y'
        ]
        self.alphabet_len = len(self.vn_alphabet)
        self.char_couples = [
            ['i', 'y'], ['s', 'x'], ['gi', 'd'],
            ['ă', 'â'], ['ch', 'tr'], ['ng', 'n'],
            ['nh', 'n'], ['ngh', 'ng'], ['ục', 'uộc'], ['o', 'u'],
            ['ă', 'a'], ['o', 'ô'], ['ả', 'ã'], ['ổ', 'ỗ'], ['ủ', 'ũ'], ['ễ', 'ể'],
            ['e', 'ê'], ['à', 'ờ'], ['ằ', 'à'], ['ẩn', 'uẩn'], ['ẽ', 'ẻ'], ['ùi', 'uồi'], ['ă', 'â'],
            ['ở', 'ỡ'], ['ỹ', 'ỷ'], ['ỉ', 'ĩ'], ['ị', 'ỵ'],
            ['ấ', 'á'], ['n', 'l'], ['qu', 'w'], ['ph', 'f'], ['d', 'z'], ['c', 'k'], ['qu', 'q'],
            ['i', 'j'], ['gi', 'j'],
        ]

        self.teencode_dict = {
            'mình': ['mk', 'mik', 'mjk'], 'vô': ['zô', 'zo', 'vo'], 'vậy': ['zậy', 'z', 'zay', 'za'],
            'phải': ['fải', 'fai', ], 'biết': ['bit', 'biet'],
            'rồi': ['rùi', 'ròi', 'r'], 'bây': ['bi', 'bay'], 'giờ': ['h', ],
            'không': ['k', 'ko', 'khong', 'hk', 'hong', 'hông', '0', 'kg', 'kh', ],
            'đi': ['di', 'dj', ], 'gì': ['j', ], 'em': ['e', ], 'được': ['dc', 'đc', ], 'tao': ['t'],
            'tôi': ['t'], 'chồng': ['ck'], 'vợ': ['vk']
        }

        self.typo = {
            "á": "as", "à": "af", "ả": "ar", "ã": "ax", "ạ": "aj",
            "ă": "aw", "ắ": "aws", "ằ": "awf", "ẳ": "awr", "ẵ": "awx", "ặ": "awj",
            "â": "aa", "ấ": "aas", "ầ": "aaf", "ẩ": "aar", "ẫ": "aax", "ậ": "aaj",
            "ó": "os", "ò": "of", "ỏ": "or", "õ": "ox", "ọ": "oj",
            "ô": "oo", "ố": "oos", "ồ": "oof", "ổ": "oor", "ỗ": "oox", "ộ": "ooj",
            "ơ": "ow", "ớ": "ows", "ờ": "owf", "ở": "owr", "ỡ": "owx", "ợ": "owj",

            "é": "es", "è": "ef", "ẻ": "er", "ẽ": "ex", "ẹ": "ej",
            "ê": "ee", "ế": "ees", "ề": "eef", "ể": "eer", "ễ": "eex", "ệ": "eej",
            "ú": "us", "ù": "uf", "ủ": "ur", "ũ": "ux", "ụ": "uj",
            "ư": "uw", "ứ": "uws", "ừ": "uwf", "ử": "uwr", "ữ": "uwx", "ự": "uwj",
            "í": "is", "ì": "if", "ỉ": "ir", "ĩ": "ix", "ị": "ij",
            "ý": "ys", "ỳ": "yf", "ỷ": "yr", "ỵ": "yj",
            "đ": "dd",

            "Á": "As", "À": "Af", "Ả": "Ar", "Ã": "Ax", "Ạ": "Aj",
            "Ă": "Aw", "Ắ": "Aws", "Ằ": "Awf", "Ẳ": "Awr", "Ẵ": "Awx", "Ặ": "Awj",
            "Â": "Aa", "Ấ": "Aas", "Ầ": "Aaf",

            "Ó": "Os", "Ò": "Of", "Ỏ": "Or", "Õ": "Ox", "Ọ": "Oj",
            "Ô": "Oo", "Ổ": "Oor", "Ỗ": "Oox", "Ộ": "Ooj", "Ố": "Oos", "Ồ": "Oof",
            "Ơ": "Ow", "Ớ": "Ows", "Ờ": "Owf", "Ở": "Owr", "Ỡ": "Owx", "Ợ": "Owj",

            "É": "Es", "È": "Ef", "Ẻ": "Er", "Ẽ": "Ex", "Ẹ": "Ej",
            "Ê": "Ee", "Ế": "Ees", "Ề": "Eef", "Ể": "Eer", "Ễ": "Eex", "Ệ": "Eej",

            "Ú": "Us", "Ù": "Uf", "Ủ": "Ur", "Ũ": "Ux", "Ụ": "Uj", "Ư": "Uw",
            "Ứ": "Uws", "Ừ": "Uwf", "Ử": "Uwr", "Ữ": "Uwx", "Ự": "Uwj",
            "Í": "Is", "Ì": "If", "Ỉ": "Ir", "Ị": "Ij", "Ĩ": "Ix",
            "Ý": "Ys", "Ỳ": "Yf", "Ỷ": "Yr", "Ỵ": "Yj",
            "Đ": "Dd"
        }

        # For VNI typing
        # self.typo = {
        #     "ă": ["aw", "a8"], "â": ["aa", "a6"], "á": ["as", "a1"], "à": ["af", "a2"], "ả": ["ar", "a3"],
        #     "ã": ["ax", "a4"], "ạ": ["aj", "a5"], "ắ": ["aws", "ă1"], "ổ": ["oor", "ô3"], "ỗ": ["oox", "ô4"],
        #     "ộ": ["ooj", "ô5"], "ơ": ["ow", "o7"],
        #     "ằ": ["awf", "ă2"], "ẳ": ["awr", "ă3"], "ẵ": ["awx", "ă4"], "ặ": ["awj", "ă5"], "ó": ["os", "o1"],
        #     "ò": ["of", "o2"], "ỏ": ["or", "o3"], "õ": ["ox", "o4"], "ọ": ["oj", "o5"], "ô": ["oo", "o6"],
        #     "ố": ["oos", "ô1"], "ồ": ["oof", "ô2"],
        #     "ớ": ["ows", "ơ1"], "ờ": ["owf", "ơ2"], "ở": ["owr", "ơ2"], "ỡ": ["owx", "ơ4"], "ợ": ["owj", "ơ5"],
        #     "é": ["es", "e1"], "è": ["ef", "e2"], "ẻ": ["er", "e3"], "ẽ": ["ex", "e4"], "ẹ": ["ej", "e5"],
        #     "ê": ["ee", "e6"], "ế": ["ees", "ê1"], "ề": ["eef", "ê2"],
        #     "ể": ["eer", "ê3"], "ễ": ["eex", "ê3"], "ệ": ["eej", "ê5"], "ú": ["us", "u1"], "ù": ["uf", "u2"],
        #     "ủ": ["ur", "u3"], "ũ": ["ux", "u4"], "ụ": ["uj", "u5"], "ư": ["uw", "u7"], "ứ": ["uws", "ư1"],
        #     "ừ": ["uwf", "ư2"], "ử": ["uwr", "ư3"], "ữ": ["uwx", "ư4"],
        #     "ự": ["uwj", "ư5"], "í": ["is", "i1"], "ì": ["if", "i2"], "ỉ": ["ir", "i3"], "ị": ["ij", "i5"],
        #     "ĩ": ["ix", "i4"], "ý": ["ys", "y1"], "ỳ": ["yf", "y2"], "ỷ": ["yr", "y3"], "ỵ": ["yj", "y5"],
        #     "đ": ["dd", "d9"],
        #     "Ă": ["Aw", "A8"], "Â": ["Aa", "A6"], "Á": ["As", "A1"], "À": ["Af", "A2"], "Ả": ["Ar", "A3"],
        #     "Ã": ["Ax", "A4"], "Ạ": ["Aj", "A5"], "Ắ": ["Aws", "Ă1"], "Ổ": ["Oor", "Ô3"], "Ỗ": ["Oox", "Ô4"],
        #     "Ộ": ["Ooj", "Ô5"], "Ơ": ["Ow", "O7"],
        #     "Ằ": ["AWF", "Ă2"], "Ẳ": ["Awr", "Ă3"], "Ẵ": ["Awx", "Ă4"], "Ặ": ["Awj", "Ă5"], "Ó": ["Os", "O1"],
        #     "Ò": ["Of", "O2"], "Ỏ": ["Or", "O3"], "Õ": ["Ox", "O4"], "Ọ": ["Oj", "O5"], "Ô": ["Oo", "O6"],
        #     "Ố": ["Oos", "Ô1"], "Ồ": ["Oof", "Ô2"],
        #     "Ớ": ["Ows", "Ơ1"], "Ờ": ["Owf", "Ơ2"], "Ở": ["Owr", "Ơ3"], "Ỡ": ["Owx", "Ơ4"], "Ợ": ["Owj", "Ơ5"],
        #     "É": ["Es", "E1"], "È": ["Ef", "E2"], "Ẻ": ["Er", "E3"], "Ẽ": ["Ex", "E4"], "Ẹ": ["Ej", "E5"],
        #     "Ê": ["Ee", "E6"], "Ế": ["Ees", "Ê1"], "Ề": ["Eef", "Ê2"],
        #     "Ể": ["Eer", "Ê3"], "Ễ": ["Eex", "Ê4"], "Ệ": ["Eej", "Ê5"], "Ú": ["Us", "U1"], "Ù": ["Uf", "U2"],
        #     "Ủ": ["Ur", "U3"], "Ũ": ["Ux", "U4"], "Ụ": ["Uj", "U5"], "Ư": ["Uw", "U7"], "Ứ": ["Uws", "Ư1"],
        #     "Ừ": ["Uwf", "Ư2"], "Ử": ["Uwr", "Ư3"], "Ữ": ["Uwx", "Ư4"],
        #     "Ự": ["Uwj", "Ư5"], "Í": ["Is", "I1"], "Ì": ["If", "I2"], "Ỉ": ["Ir", "I3"], "Ị": ["Ij", "I5"],
        #     "Ĩ": ["Ix", "I4"], "Ý": ["Ys", "Y1"], "Ỳ": ["Yf", "Y2"], "Ỷ": ["Yr", "Y3"], "Ỵ": ["Yj", "Y5"],
        #     "Đ": ["Dd", "D9"]
        # }

        self.all_word_candidates = self.get_all_word_candidates()
        self.string_all_word_candidates = ' '.join(self.all_word_candidates)
        self.all_char_candidates = self.get_all_char_candidates()
        self.keyboard_neighbors = self.get_keyboard_neighbors()

    def replace_teencode(self, word):
        candidates = self.teencode_dict.get(word, None)
        if candidates is not None:
            chosen_one = 0
            if len(candidates) > 1:
                chosen_one = np.random.randint(0, len(candidates))
            return candidates[chosen_one]

    def get_keyboard_neighbors(self):
        keyboard_neighbors = {}
        keyboard_neighbors['a'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ă'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['â'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['á'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['à'] = "aáàảãăắằẳẵâấầẩẫ"
        keyboard_neighbors['ả'] = "aảã"
        keyboard_neighbors['ã'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ạ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ắ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ằ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ẳ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ặ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ẵ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ấ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ầ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ẩ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ẫ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['ậ'] = "aáàảãạăắằẳẵặâấầẩẫậ"
        keyboard_neighbors['b'] = "bh"
        keyboard_neighbors['c'] = "cgn"
        keyboard_neighbors['d'] = "đctơở"
        keyboard_neighbors['đ'] = "d"
        keyboard_neighbors['e'] = "eéèẻẽẹêếềểễệbpg"
        keyboard_neighbors['é'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['è'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['ẻ'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['ẽ'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['ẹ'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['ê'] = "eéèẻẽẹêếềểễệá"
        keyboard_neighbors['ế'] = "eéèẻẽẹêếềểễệố"
        keyboard_neighbors['ề'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['ể'] = "eéèẻẽẹêếềểễệôốồổỗộ"
        keyboard_neighbors['ễ'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['ệ'] = "eéèẻẽẹêếềểễệ"
        keyboard_neighbors['g'] = "qgộ"
        keyboard_neighbors['h'] = "hgj"
        keyboard_neighbors['i'] = "iíìỉĩịat"
        keyboard_neighbors['í'] = "iíìỉĩị"
        keyboard_neighbors['ì'] = "iíìỉĩị"
        keyboard_neighbors['ỉ'] = "iíìỉĩị"
        keyboard_neighbors['ĩ'] = "iíìỉĩị"
        keyboard_neighbors['ị'] = "iíìỉĩịhự"
        keyboard_neighbors['k'] = "klh"
        keyboard_neighbors['l'] = "ljidđ"
        keyboard_neighbors['m'] = "mn"
        keyboard_neighbors['n'] = "mnedư"
        keyboard_neighbors['o'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ó'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ò'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ỏ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['õ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ọ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ô'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ố'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ồ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ổ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ộ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ỗ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ơ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ớ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ờ'] = "oóòỏọõôốồổỗộơớờởợỡà"
        keyboard_neighbors['ở'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ợ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        keyboard_neighbors['ỡ'] = "oóòỏọõôốồổỗộơớờởợỡ"
        # keyboard_neighbors['p'] = "op"
        # keyboard_neighbors['q'] = "qọ"
        # keyboard_neighbors['r'] = "rht"
        # keyboard_neighbors['s'] = "s"
        # keyboard_neighbors['t'] = "tp"
        keyboard_neighbors['u'] = "uúùủũụưứừữửựhiaạt"
        keyboard_neighbors['ú'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ù'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ủ'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ũ'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ụ'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ư'] = "uúùủũụưứừữửựg"
        keyboard_neighbors['ứ'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ừ'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ử'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ữ'] = "uúùủũụưứừữửự"
        keyboard_neighbors['ự'] = "uúùủũụưứừữửựg"
        keyboard_neighbors['v'] = "v"
        keyboard_neighbors['x'] = "x"
        keyboard_neighbors['y'] = "yýỳỷỵỹụ"
        keyboard_neighbors['ý'] = "yýỳỷỵỹ"
        keyboard_neighbors['ỳ'] = "yýỳỷỵỹ"
        keyboard_neighbors['ỷ'] = "yýỳỷỵỹ"
        keyboard_neighbors['ỵ'] = "yýỳỷỵỹ"
        keyboard_neighbors['ỹ'] = "yýỳỷỵỹ"
        # keyboard_neighbors['w'] = "wv"
        # keyboard_neighbors['j'] = "jli"
        # keyboard_neighbors['z'] = "zs"
        # keyboard_neighbors['f'] = "ft"

        return keyboard_neighbors

    def replace_char_noaccent(self, text, onehot_label):
        # find index noise
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        index_noise = idx
        # onehot_label[index_noise] = 1
        word_noise = text[index_noise]
        for i in range(0, len(word_noise)):
            char = word_noise[i]

            if char in self.keyboard_neighbors:
                neighbors = self.keyboard_neighbors[char]
                idx_neigh = np.random.randint(0, len(neighbors))
                replaced = neighbors[idx_neigh]
                new_word = word_noise[:i] + replaced + word_noise[i + 1:]
                if new_word == word_noise:
                    continue

                text[index_noise] = new_word
                onehot_label[index_noise] = 1
                return True, text, onehot_label

        return False, text, onehot_label

    def replace_word_candidate(self, word):
        """
        Return a homophone word of the input word.
        """
        capital_flag = word[0].isupper()
        word = word.lower()
        if capital_flag and word in self.teencode_dict:
            return self.replace_teencode(word).capitalize()
        elif word in self.teencode_dict:
            return self.replace_teencode(word)

        for couple in self.word_couples:
            for i in range(2):
                if couple[i] == word:
                    if i == 0:
                        if capital_flag:
                            return couple[1].capitalize()
                        else:
                            return couple[1]
                    else:
                        if capital_flag:
                            return couple[0].capitalize()
                        else:
                            return couple[0]

    def replace_char_candidate(self, char):
        """
        return a homophone char/subword of the input char.
        """
        for couple in self.char_couples:
            for i in range(2):
                if couple[i] == char:
                    if i == 0:
                        return couple[1]
                    else:
                        return couple[0]

    def replace_char_candidate_typo(self, char: str):
        """
        return a homophone char/subword of the input char.
        """
        i = np.random.randint(0, 2)

        # return self.typo[char][i]
        return self.typo[char]

    def get_all_char_candidates(self):
        all_char_candidates = []
        for couple in self.char_couples:
            all_char_candidates.extend(couple)
        return all_char_candidates

    def get_all_word_candidates(self):
        all_word_candidates = []
        for couple in self.word_couples:
            all_word_candidates.extend(couple)
        return all_word_candidates

    def remove_diacritics(self, text, onehot_label):
        """
        Replace word which has diacritics with the same word without diacritics
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modified, so this
            function only choose the word that does not have onehot label == 1.
        return: a list of word tokens has one word that its diacritics was removed,
                a list of onehot label indicate the position of words that has been modified.
        """
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx] == unidecode.unidecode(text[idx]) or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        onehot_label[idx] = 1
        text[idx] = unidecode.unidecode(text[idx])
        return True, text, onehot_label

    def replace_with_random_letter(self, text, onehot_label):
        """
        Replace, add (or remove) a random letter in a random chosen word with a random letter
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modified, so this
            function only choose the word that does not have onehot label == 1.
        return: a list of word tokens has one word that has been modified,
                a list of onehot label indicate the position of words that has been modified.
        """
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        # replace, add or remove? 0 is replaced, 1 is added, 2 is removed
        coin = np.random.choice([0, 1, 2])
        if coin == 0:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            replaced = self.vn_alphabet[np.random.randint(0, self.alphabet_len)]
            try:
                text[idx] = re.sub(chosen_letter, replaced, text[idx])
            except:
                return False, text, onehot_label
        elif coin == 1:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            replaced = chosen_letter + self.vn_alphabet[np.random.randint(0, self.alphabet_len)]
            try:
                text[idx] = re.sub(chosen_letter, replaced, text[idx])
            except:
                return False, text, onehot_label
        else:
            chosen_letter = text[idx][np.random.randint(0, len(text[idx]))]
            try:
                text[idx] = re.sub(chosen_letter, '', text[idx])
            except:
                return False, text, onehot_label

        onehot_label[idx] = 1
        return True, text, onehot_label

    def replace_with_homophone_word(self, text, onehot_label):
        """
        Replace a candidate word (if exist in the word_couple) with its homophone.
        If succeeded, return True, else False

        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modified, so this
            function only choose the word that does not have onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, text, onehot_label
        """
        candidates = []
        for i in range(len(text)):
            # account for the case that the word in the text is upper case but its lowercase match the candidates list
            if text[i].lower() in self.all_word_candidates or text[i].lower() in self.teencode_dict.keys():
                candidates.append((i, text[i]))

        if len(candidates) == 0:
            return False, text, onehot_label

        idx = np.random.randint(0, len(candidates))
        prevent_loop = 0
        while onehot_label[candidates[idx][0]] == 1:
            idx = np.random.choice(np.arange(0, len(candidates)))
            prevent_loop += 1
            if prevent_loop > 5:
                return False, text, onehot_label

        text[candidates[idx][0]] = self.replace_word_candidate(candidates[idx][1])
        onehot_label[candidates[idx][0]] = 1
        return True, text, onehot_label

    def replace_with_homophone_letter(self, text, onehot_label):
        """
        Replace a subword/letter with its homophones
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modified, so this
            function only choose the word that does not have onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, None, None
        """
        candidates = []
        for i in range(len(text)):
            for char in self.all_char_candidates:
                if re.search(char, text[i]) is not None:
                    candidates.append((i, char))
                    break

        if len(candidates) == 0:
            return False, text, onehot_label
        else:
            idx = np.random.randint(0, len(candidates))
            prevent_loop = 0
            while onehot_label[candidates[idx][0]] == 1:
                idx = np.random.randint(0, len(candidates))
                prevent_loop += 1
                if prevent_loop > 5:
                    return False, text, onehot_label

            replaced = self.replace_char_candidate(candidates[idx][1])
            text[candidates[idx][0]] = re.sub(candidates[idx][1], replaced, text[candidates[idx][0]])

            onehot_label[candidates[idx][0]] = 1
            return True, text, onehot_label

    def replace_with_typo_letter(self, text, onehot_label):
        """
        Replace a subword/letter with its homophones
        Args:
            text: a list of word tokens
            onehot_label: onehot array indicate position of word that has already modified, so this
            function only choose the word that does not have onehot label == 1.
        return: True, text, onehot_label if successful replace, else False, None, None
        """
        # find index noise
        idx = np.random.randint(0, len(onehot_label))
        prevent_loop = 0
        while onehot_label[idx] == 1 or text[idx].isnumeric() or text[idx] in string.punctuation:
            idx = np.random.randint(0, len(onehot_label))
            prevent_loop += 1
            if prevent_loop > 10:
                return False, text, onehot_label

        index_noise = idx
        word_noise = text[index_noise]

        for j in range(0, len(word_noise)):
            char = word_noise[j]

            if char in self.typo:
                replaced = self.replace_char_candidate_typo(char)
                word_noise = word_noise[:j] + replaced + word_noise[j + 1:]
                text[index_noise] = word_noise

                onehot_label[index_noise] = 1
                return True, text, onehot_label

        return False, text, onehot_label

    def add_noise(self, sentence, percent_err=0.15, num_type_err=5):
        tokens = self.tokenizer(sentence)
        onehot_label = [0] * len(tokens)

        num_wrong = int(np.ceil(percent_err * len(tokens)))
        num_wrong = np.random.randint(1, num_wrong + 1)
        if np.random.rand() < 0.05:
            num_wrong = 0

        for i in range(0, num_wrong):
            err = np.random.randint(0, num_type_err + 1)

            if err == 0:
                _, tokens, onehot_label = self.replace_with_homophone_letter(tokens, onehot_label)
            elif err == 1:
                _, tokens, onehot_label = self.replace_with_typo_letter(tokens, onehot_label)
            elif err == 2:
                _, tokens, onehot_label = self.replace_with_homophone_word(tokens, onehot_label)
            elif err == 3:
                _, tokens, onehot_label = self.replace_with_random_letter(tokens, onehot_label)
            elif err == 4:
                _, tokens, onehot_label = self.remove_diacritics(tokens, onehot_label)
            elif err == 5:
                _, tokens, onehot_label = self.replace_char_noaccent(tokens, onehot_label)
            else:
                continue
            # print(tokens)
        return ' '.join(tokens)


if __name__ == '__main__':
    # np.random.seed(123)

    txt = "Khoảng 13h30 ngày 7/6, nhiều ôtô biển xanh ra vào trụ sở Bộ Y tế trên phố Giảng Võ, chắc 10.2%" \
          "phường Kim Mã, quận Ba Đình, Hà Nội. Đến khoảng 14h20 cùng ngày (ngày 14/2), một cán bộ cảnh sát " \
          "ngồi trong chiếc xe biển xanh -\"công an\" tiếp tục đi vào trụ sở của cơ quan này."
    tks = word_tokenize(txt)
    print(tks)

    synthesizer = SynthesizeData()

    count = 0
    for _ in range(100):
        tks = word_tokenize(txt)
        _, tks, onehot = synthesizer.replace_with_typo_letter(tks, [0] * len(tks))
        if sum(onehot) > 0:
            count += 1

    print(f"Total: {count / 100:.5f}")

    print(' '.join(tks))
    print(onehot)

    # for z in range(len(onehot)):
    #     if onehot[z] == 1:
    #         print(tks[z])
