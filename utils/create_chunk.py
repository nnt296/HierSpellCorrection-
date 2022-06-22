import json
import os

import pandas as pd
import regex
# import underthesea as uts

from cleantext import remove_emoji

pattern = regex.compile(r"""
[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ
fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRs
StTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-.
/:;<=>?@[\]^_`{|}~[:space:]]""")

if __name__ == '__main__':
    lines_per_file = pow(2, 12)

    root_dir = "/home/local/BM/Datasets/SpellNews/val"
    file_idx = 0
    file_name = os.path.join(root_dir, f"corpus_{file_idx}.txt")
    num_line = 0

    big_corpus = "/home/local/BM/Datasets/BM/test.csv"

    chunk_size = 10 ** 4
    with pd.read_csv(big_corpus, chunksize=chunk_size) as reader:
        fp = open(file_name, "a", encoding="utf-8")

        for chunk in reader:
            for _, row in chunk.iterrows():
                tdb = [row["Title"], row["Description"], row["Body"]]

                for text in tdb:
                    if pd.isna(text):
                        continue

                    # Remove newline
                    text = text.replace("\r", "").replace("\n", " ")
                    # Remove weird space char
                    text = text.replace('\u200b', '')
                    text = remove_emoji(text)
                    text = pattern.sub("", text).strip()
                    if not text:
                        continue

                    paragraphs = text.split("\n")
                    for par in paragraphs:
                        par = par.strip()
                        if not par:
                            continue

                        fp.write(f"{par}\n")
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
