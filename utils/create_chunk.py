import json
import os
import regex
# import underthesea as uts

from cleantext import remove_emoji

pattern = regex.compile(r"""
[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ
fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRs
StTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-.
/:;<=>?@[\]^_`{|}~[:space:]]""")

if __name__ == '__main__':
    lines_per_file = pow(2, 15)

    root_dir = "/home/chungvv/thanhnn5/datasets/SpellNews/train"
    file_idx = 0
    file_name = os.path.join(root_dir, f"corpus_{file_idx}.txt")
    num_line = 0

    big_corpus = "./data/corpus_big.txt"

    with open(big_corpus) as reader:
        fp = open(file_name, "a", encoding="utf-8")
        while True:
            text = reader.readline()
            if not text:
                break

            # Remove newline
            text = text.replace("\r", "").replace("\n", " ")
            # Remove weird space char
            text = text.replace('\u200b', '')
            text = remove_emoji(text)
            text = pattern.sub("", text).strip()
            text = text.replace('\n', ' ')
            if not text:
                continue

            fp.write(f"{text}\n")
            num_line += 1
            if num_line % lines_per_file == 0:
                fp.close()
                file_idx += 1
                file_name = os.path.join(root_dir, f"corpus_{file_idx}.txt")
                fp = open(file_name, "a", encoding="utf-8")
    fp.close()

    stats_file = os.path.join(root_dir, "stats.json")
    stats = {
        "lines_per_file": lines_per_file,
        "num_files": len(os.listdir(root_dir)),
        "total_lines": num_line
    }
    json.dump(stats, open(stats_file, "w"), indent=2)
