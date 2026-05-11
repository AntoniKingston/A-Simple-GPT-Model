import os
import wget
def download_txt(download_path, catalog_url, unwanted_titles):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    catalog_path = download_path + "/catalog.html"
    if not os.path.exists(catalog_path):
        wget.download(catalog_url, out=catalog_path)
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
        url = url_prefix + title + url_suffix
        text_path = download_path + "/" + title + ".txt"
        if not os.path.exists(text_path):
            # Sometimes .txt is not available, so we skip the download
            try:
                wget.download(url, out = text_path)
            except:
                print("Error downloading " + title)


def merge_txt(dir_path, out_name):
    out_path = dir_path + "/" + out_name
    out_text = ""
    work_sep = "\n\n\n"
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt"):
            with open(dir_path + "/" + file_name, "r") as f:
                out_text += f.read() + work_sep
    out_text = out_text[:-len(work_sep)]
    f = open(out_path, "w")
    f.write(out_text)
    f.close()

if __name__ == "__main__":
    download_txt("data", "https://wolnelektury.pl/katalog/autor/adam-mickiewicz/", ['mickiewicius-baltas-karzigys',
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
    merge_txt("data", "mickiewicz_merged.txt")
