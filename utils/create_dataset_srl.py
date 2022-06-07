import sys
from this import s
from underthesea import sent_tokenize, word_tokenize
import random
import numpy as np
import re
from tqdm import tqdm
import string 
import locale

# Constants
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

s3 = u'ẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẾếỀềỂểỄễỆệỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỨứỪừỬửỮữỰự'
s2 = u'ÂâÂâÂâÂâÂâĂăĂăĂăĂăĂăÊêÊêÊêÊêÊêÔôÔôÔôÔôÔôƠơƠơƠơƠơƠơƯưƯưƯưƯưƯư'
alphabet = 'abcdefghijklmnopqrstuvwxyz'

s5 = ['úy', 'ùy', 'ủy', 'ũy', 'ụy', 'óa', 'òa', 'ỏa', 'õa', 'ọa']
s4 = ['uý', 'uỳ', 'uỷ', 'uỹ', 'uỵ', 'oá', 'oà', 'oả', 'oã', 'oạ']

vowels = ['a', 'á', 'à', 'ả', 'ạ', 'ă', 'ắ', 'ằ', 'ẳ', 'ặ', 'â', 'ấ', 'ầ', 'ẩ', 'ậ', 
        'e', 'é', 'è', 'ẻ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ệ', 'i', 'í', 'ì', 'ỉ', 'ị', 
        'o', 'ó', 'ò', 'ỏ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ợ', 
        'u', 'ú', 'ù', 'ủ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ự', 'y', 'ý', 'ỳ', 'ỷ', 'ỵ']

def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def remove_special_char(input_article):
    pattern = ">>.+>>" # remove hyperlinks between ">>"

    sentences = sent_tokenize(input_article) # sentence tokenize
    for i in range(len(sentences)):
        sentences[i] = re.sub(pattern,"",sentences[i])
        sentences[i] = sentences[i].replace(".. ",". ")
    sentences = [" ".join(word_tokenize(sent)) for sent in sentences]
    return sentences

def read_raw_text(input_file):
    with open(input_file, 'r') as fr:
        text = fr.read()
    return text

def read_keywords_file(keywords_file):
    with open(keywords_file, 'r') as fr:
        keywords = fr.readlines()
    return keywords

def find_articles_by_keyword(articles, keywords):
    articles_by_keywords = []
    for article in articles:
        if article.find(keywords):
            pass

# check if a token is number or not (e.g. 15.20 or 15,20)
def is_number(string):
    try:
        float(string) 
        return True
    except ValueError:
        pass
    try:
        locale.atoi(string)
        return True
    except ValueError:
        pass
    return False

# check if a token is hour or measurement (e.g. 15h20, 3h, 1m55)    
def is_time_or_measurement(string):
    time_pattern = r'\dh\d{2}'
    time_pattern2 = r'\dh'
    measurement_pattern = r'\dm\d'
    time_search = re.search(time_pattern, string)
    time_search2 = re.search(time_pattern2, string)
    measurement_search = re.search(measurement_pattern, string)
    return time_search or time_search2 or measurement_search

# check if a token include 'uy' or 'oa' 
def special_tone_contain(string):
  for i in range(len(s5)):
    if string.find(s5[i]) > 0:
      return i
  return False

# use for "ngong" cases
def manual_replace(s, char, index, length):
    return s[:index] + char + s[index +length:]

# main function
def generate_typos_and_pos(token,
                   no_typo_prob=0.6,
                   ngong_typo_prob=0.4,
                   special_tone_prob=0.3,
                   asccents_prob=0.3,
                   swap_char_prob=0.1,
                   add_chars_prob=0.2,
                   remove_chars_prob=0.2
                   ):
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    pos = "C"
    typo_type = "None"
    if is_number(token) or is_time_or_measurement(token) or token in string.punctuation: # skip typo on digit 
        return token, pos, typo_type
    if random.random() < no_typo_prob:
        # print("No typo prob")
        return token, pos, typo_type
    # Tha dau "oa", "uy"
    if special_tone_contain(token) and random.random() < special_tone_prob:
        special_tone_index = special_tone_contain(token)
        token = token.replace(s5[special_tone_index], s4[special_tone_index])
        pos = "TYPO"
        typo_type = "Special tone swap"
        return token, pos, typo_type
    # Tha dau "oa", "uy"

    # Ngong l-n
    if token.startswith('l') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'n', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong l-n"
        return token, pos, typo_type
    if token.startswith('L') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'N', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong l-n"
        return token, pos, typo_type    
    # Ngong n-l
    if token.startswith('n') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'l', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong n-l"
        return token, pos, typo_type
    if token.startswith('N') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'L', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong n-l"
        return token, pos, typo_type
    # Ngong s-x
    if token.startswith('s') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'x', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong s-x"
        return token, pos, typo_type
    if token.startswith('S') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'X', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong s-x"
        return token, pos, typo_type
    # Ngong x-s
    if token.startswith('x') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 's', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong x-s"
        return token, pos, typo_type
    if token.startswith('X') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'S', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong x-s"
        return token, pos, typo_type
    # Ngong tr-ch
    if token.startswith('tr') and len(token) > 3 and token[2] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'ch', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong tr-ch"
        return token, pos, typo_type
    if token.startswith('Tr') and len(token) > 3 and token[2] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'Ch', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong tr-ch"
        return token, pos, typo_type
    # Ngong ch-tr
    if token.startswith('ch') and len(token) > 3 and token[2] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'tr', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong ch-tr"
        return token, pos, typo_type
    if token.startswith('Ch') and len(token) > 3 and token[2] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'Tr', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong ch-tr"
        return token, pos, typo_type
    # Ngong gi-d
    if token.startswith('gi') and len(token) > 3 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'd', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong gi-d"
        return token, pos, typo_type
    if token.startswith('Gi') and len(token) > 3 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'D', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong gi-d"
        return token, pos, typo_type
    # Ngong d-gi
    if token.startswith('d') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'gi', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong d-gi"
        return token, pos, typo_type
    if token.startswith('D') and len(token) > 3 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'Gi', 0, 2)
        pos = "TYPO"
        typo_type = "Ngong d-gi"
        return token, pos, typo_type
    # Ngong r-d
    if token.startswith('r') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'd', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong r-d"
        return token, pos, typo_type
    if token.startswith('R') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'D', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong r-d"
        return token, pos, typo_type
    # Ngong d-r
    if token.startswith('d') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'r', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong d-r"
        return token, pos, typo_type
    if token.startswith('D') and len(token) > 2 and token[1] in vowels and random.random() < ngong_typo_prob:
        token = manual_replace(token, 'R', 0, 1)
        pos = "TYPO"
        typo_type = "Ngong d-r"
        return token, pos, typo_type

    # Remove accent
    if random.random() < asccents_prob:
        if random.random() < 0.5:
            new_token = remove_accents(token)
        else:
            new_chars = []
            for cc in token:
                if cc in s3 and random.random() < 0.7:
                    cc = s2[s3.index(cc)]
                if cc in s1 and random.random() < 0.5:
                    cc = s0[s1.index(cc)]                    
                new_chars.append(cc)
            new_token = "".join(new_chars)
        
        if new_token != token:
            pos = "TYPO"
            typo_type = "Remove accent"
            token = new_token
            
    # if random.random() < lowercase_prob:
    #     print("lowercase_prob")
    #     token = token.lower()
    #     # print(token)
    if random.random() < swap_char_prob:
        chars = list(token)
        n_swap = min(len(chars), np.random.poisson(0.5) + 1)
        index = np.random.choice(
            np.arange(len(chars)), size=n_swap, replace=False)
        swap_index = index[np.random.permutation(index.shape[0])]
        swap_dict = {ii: jj for ii, jj in zip(index, swap_index)}
        chars = [chars[ii] if ii not in index else chars[swap_dict[ii]]
                 for ii in range(len(chars))]
        new_token = "".join(chars)
        if new_token != token: 
            token = new_token
            pos = "TYPO"
            typo_type = "Swap char"
    if random.random() < remove_chars_prob:
        n_remove = min(len(token), np.random.poisson(0.005) + 1)
        for _ in range(n_remove):
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = token[:pos] + token[pos+1:]
        pos = "TYPO"
        typo_type = "Remove char"
    if random.random() < add_chars_prob:
        n_add = min(len(token), np.random.poisson(0.05) + 1)
        adding_chars = np.random.choice(
            list(alphabet), size=n_add, replace=True)
        for cc in adding_chars:
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = "".join([token[:pos], cc, token[pos:]])
        pos = "TYPO"
        typo_type = "Add char"

    return token, pos, typo_type

# input a sentence and return a list of tokens, typo position and typo type (optional)
def generate_typos_and_pos_for_text(sentence):
    new_texts = []
    typo_pos = []
    typo_types = []
    
    if len(sentence) > 0:
        for s in sentence:
            new_tokens = []
            for t in s.split():                
                new_t, typo_label, typo_type = generate_typos_and_pos(t)
                new_tokens.append(new_t)
                typo_pos.append(typo_label)
                typo_types.append(typo_type)
            new_sentence = " ".join(new_tokens)

            new_texts.append(new_sentence)

    return new_texts, typo_pos, typo_types

def main():
    input_file = sys.argv[1] 
    train_output = sys.argv[2]
    test_output = sys.argv[3]   

    print("Reading input file")
    text = read_raw_text(input_file)
    articles = text.split("\n")
    print("Number of articles: ", len(articles))
    train_samples = int(len(articles)*0.8)

    with open(train_output, "w") as fw:        
        sentence_idx = 0

        for a in tqdm(articles[:train_samples]):
            preprocessed_sentences = remove_special_char(a)
            for sentence in preprocessed_sentences:       
                # if sentence is too short, skip
                if len(sentence.split()) > 2: 
                    sentence_idx = sentence_idx + 1
                    typo_sentence, pos, typo_types = generate_typos_and_pos_for_text([sentence])
                    if len(typo_sentence[0].split()) == len(pos):
                        for i in range(len(pos)):                            
                            # fw.write(typo_sentence[0].split()[i] + '\t' + pos[i] + '\t' + typo_types[i] + '\n')
                            fw.write(typo_sentence[0].split()[i] + '\t' + pos[i] + '\n')
                fw.write('\n')
    print("Number of train sentences: ", sentence_idx)
    with open(test_output, "w") as fw:        
        sentence_idx = 0

        for a in tqdm(articles[train_samples+1:]):
            preprocessed_sentences = remove_special_char(a)
            for sentence in preprocessed_sentences:       
                # if sentence is too short, skip
                if len(sentence.split()) > 2: 
                    sentence_idx = sentence_idx + 1
                    typo_sentence, pos, typo_types = generate_typos_and_pos_for_text([sentence])
                    if len(typo_sentence[0].split()) == len(pos):
                        for i in range(len(pos)):                            
                            # fw.write(typo_sentence[0].split()[i] + '\t' + pos[i] + '\t' + typo_types[i] + '\n')
                            fw.write(typo_sentence[0].split()[i] + '\t' + pos[i] + '\n')
                fw.write('\n')
    print("Number of test sentences: ", sentence_idx)
    print("Done")  

if __name__=="__main__":
    main()
