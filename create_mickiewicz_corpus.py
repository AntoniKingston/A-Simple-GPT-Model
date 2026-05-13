import os
import wget
import re

from numpy.f2py.auxfuncs import throw_error
from streamlit.runtime import fragment


def check_for_phrase(lines, phrase):
    for i, line in enumerate(lines):
        if phrase in line:
            return i, True
    return -1, False
def download_single_txt(title, url_prefix="https://wolnelektury.pl/media/book/txt/", file_name_suffix=".txt", data_folder="data/separate_txts"):
    file_name = title + file_name_suffix
    url = url_prefix + file_name
    out_path = os.path.join(data_folder, file_name)
    if not os.path.exists(out_path):
        print(f"Downloading {title} from {url} to {data_folder}")
        try:
            wget.download(url, out=data_folder, bar=None)
        except:
            print("\nError downloading " + title)
            # print(url_prefix)
            # print(file_name_suffix)
            # print(data_folder)
            # print(file_name)
            # print(url)


def extract_tag_nesting(lines, start_line):
    opened_ctr = 0
    enclosed_lines = []
    start_regex = r"<[^/]"
    end_regex = r"</"
    break_flag = False
    for line in lines[start_line:]:
        pos_matches = re.finditer(start_regex, line)
        pos_positions = [m.start() for m in pos_matches]
        neg_matches = re.finditer(end_regex, line)
        neg_positions = [m.start() for m in neg_matches]
        while pos_positions or neg_positions:
            if (not neg_positions) or (pos_positions and pos_positions[0] < neg_positions[0]):
                opened_ctr += 1
                pos_positions.pop(0)
            elif (not pos_positions) or (neg_positions and neg_positions[0] < pos_positions[0]):
                opened_ctr -= 1
                neg_positions.pop(0)
            if opened_ctr == 0:
                break_flag = True
                break
        enclosed_lines.append(line)
        if break_flag:
            break
    return enclosed_lines
# Total spaghetti, but it works
def extract_deepest_tag_nesting(lines):
    opened_ctrs = []
    start_regex = r"<[^/]"
    end_regex = r"</"
    opened_ctr = 0
    max_depth = -1
    first_max_depth_pos = 0
    for id, line in enumerate(lines):
        pos_matches = re.finditer(start_regex, line)
        pos_positions = [m.start() for m in pos_matches]
        neg_matches = re.finditer(end_regex, line)
        neg_positions = [m.start() for m in neg_matches]
        while pos_positions or neg_positions:
            if (not neg_positions) or (pos_positions and pos_positions[0] < neg_positions[0]):
                opened_ctr += 1
                pos_positions.pop(0)
            elif (not pos_positions) or (neg_positions and neg_positions[0] < pos_positions[0]):
                opened_ctr -= 1
                neg_positions.pop(0)
            if opened_ctr > max_depth:
                max_depth = opened_ctr
                first_max_depth_pos = len(opened_ctrs)
            opened_ctrs.append((opened_ctr, id))
    second_max_depth_pos = 0
    for i, (depth, id) in enumerate(opened_ctrs[first_max_depth_pos+1:]):
        if depth == max_depth:
            second_max_depth_pos = i + first_max_depth_pos + 1
            break
        if second_max_depth_pos == 0:
            Exception("No second max depth found")
    min_between_maxes = 999999
    for i, (depth, id) in enumerate(opened_ctrs[first_max_depth_pos+1:second_max_depth_pos]):
        if depth < min_between_maxes:
            min_between_maxes = depth
    last_min_before_first_max_pos = 0
    for i, (depth, id) in enumerate(opened_ctrs):
        if depth == min_between_maxes + 1:
            last_min_before_first_max_pos = i
        if depth == max_depth:
            break
    last_max_pos = 0
    for i, (depth, id) in enumerate(opened_ctrs[second_max_depth_pos+1:]):
        if depth  == max_depth:
            last_max_pos = i + second_max_depth_pos + 1
    first_min_after_last_max_pos = 0
    for i, (depth, id) in enumerate(opened_ctrs[last_max_pos+1:]):
        if depth == min_between_maxes:
            first_min_after_last_max_pos = i + last_max_pos + 1
            break
    # print(first_max_depth_pos, second_max_depth_pos, last_max_pos, min_between_maxes, last_min_before_first_max_pos, first_min_after_last_max_pos, sep='\n')
    _, real_start = opened_ctrs[last_min_before_first_max_pos]
    _, real_end = opened_ctrs[first_min_after_last_max_pos]
    return lines[real_start:(real_end+1)]


def extract_phrases(lines, match_phrase, sep, pos):
    phrases = []
    for line in lines:
        if match_phrase in line:
            splits = line.split(sep)
            wanted_fragment = splits[pos]
            phrases.append(wanted_fragment)
    return phrases

def scan_html_recursive(title,
                        html_url_prefix="https://wolnelektury.pl/katalog/lektura/",
                        txt_url_prefix = "https://wolnelektury.pl/media/book/txt/",
                        html_name_suffix = ".html",
                        txt_name_suffix = ".txt",
                        data_folder = "data",
                        txt_folder = "separate_txts",
                        consideration_phrase="l-aside__zbiory",
                        forbidden_phrase = "strong"):
    txt_data_folder = os.path.join(data_folder, txt_folder)
    download_single_txt(title, txt_url_prefix, txt_name_suffix, txt_data_folder)
    file_name = title + html_name_suffix
    html_path = os.path.join(data_folder, file_name)
    if not os.path.exists(html_path):
        url = html_url_prefix + title
        wget.download(url, out=html_path, bar=None)
    with open(html_path, "r") as f:
        lines = f.readlines()
    os.remove(html_path)
    consideration_phrase_line_id, return_flag = check_for_phrase(lines, consideration_phrase)
    if not return_flag:
        return
    consideration_phrase_tag_nesting = extract_tag_nesting(lines, consideration_phrase_line_id)
    consideration_phrase_deepest_tag_nesting = extract_deepest_tag_nesting(consideration_phrase_tag_nesting)
    # # for debug purposes
    # if title == "dziady-dziadow-czesci-iii-ustep":
    #     print("".join(consideration_phrase_tag_nesting))
    #     print("".join(consideration_phrase_deepest_tag_nesting))
    _, return_flag = check_for_phrase(consideration_phrase_deepest_tag_nesting, forbidden_phrase)
    if return_flag:
        return
    titles = extract_phrases(consideration_phrase_deepest_tag_nesting, "<a href", "/", 3)
    for subtitle in titles:
        scan_html_recursive(subtitle, html_url_prefix, txt_url_prefix, html_name_suffix, txt_name_suffix, data_folder, txt_folder, consideration_phrase, forbidden_phrase)
def download_corpus(catalog_url,
                    unwanted_titles,
                    out_name = "mickiewicz_corpus.txt",
                    download_path = "data",
                    temp_folder = "separate_txts",
                    html_url_prefix="https://wolnelektury.pl/katalog/lektura/",
                    txt_url_prefix = "https://wolnelektury.pl/media/book/txt/",
                    html_name_suffix = ".html",
                    txt_name_suffix = ".txt",
                    consideration_phrase="l-aside__zbiory",
                    forbidden_phrase = "strong",
                    meta_info_sep = "-----",
                    remove_after_merge=False):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(os.path.join(download_path, temp_folder)):
        os.makedirs(os.path.join(download_path, temp_folder))
    catalog_path = os.path.join(download_path, "catalog.html")
    if not os.path.exists(catalog_path):
        wget.download(catalog_url, out=catalog_path)
        print("\n")
    titles = []
    with open(catalog_path, "r") as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        if line.startswith("EOF"):
            break
        if line.startswith('<article class="l-books__item book-container-activator"'):
            i = 3
        if line.startswith('    <a href="/katalog/lektura/') and i > 0:
            titles.append(line.split('/')[-2])
        i -= 1
    # We want only works written in polish
    for title in unwanted_titles:
        titles.remove(title)
    url_prefix = "https://wolnelektury.pl/media/book/txt/"
    url_suffix = ".txt"
    for title in titles:
        text_path = os.path.join(download_path, title + ".txt")
        if not os.path.exists(text_path):
            download_single_txt(title, url_prefix, url_suffix, os.path.join(download_path, temp_folder))
        scan_html_recursive(title,
                            html_url_prefix,
                            txt_url_prefix,
                            html_name_suffix,
                            txt_name_suffix,
                            download_path,
                            temp_folder,
                            consideration_phrase,
                            forbidden_phrase)
    merge_txt(temp_folder, download_path, out_name, meta_info_sep, remove_after_merge)



def merge_txt(in_dir_name, out_dir_name, out_file_name, meta_info_sep = "-----", remove_after_merge=True):
    out_dir_path = out_dir_name
    out_path = os.path.join(out_dir_path, out_file_name)
    out_text = ""
    in_dir_path = os.path.join(out_dir_path, in_dir_name)
    for file_name in os.listdir(in_dir_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(in_dir_path, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()[6:]
                cur_corpus = "".join(lines).split(meta_info_sep)[0]
                out_text += (cur_corpus)
            if remove_after_merge:
                os.remove(file_path)
    f = open(out_path, "w")
    f.write(out_text)
    f.close()

if __name__ == "__main__":
    download_corpus("https://wolnelektury.pl/katalog/autor/adam-mickiewicz/", ['mickiewicius-baltas-karzigys',
                                                                                    'mickievicius-grazina',
                                                                                    'mickievicius-konradas-valenrodas',
                                                                                    'krymo-sonetai',
                                                                                    'mickievicius-nemunui',
                                                                                    'mickievicius-ponia-tvardauskiene',
                                                                                    'mickiewicz-the-tempest',
                                                                                    'mickiewicz-to-my-cicerone',
                                                                                    'mickiewicz-to-upon-the-alps-in-splugen-1829',
                                                                                    'mickiewicz-uncertainty',
                                                                                    'mickevicius-vaidilos-apysaka',
                                                                                    'mickievicius-vilija',
                                                                                    'giaur'])
