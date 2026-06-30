import os
import sys
import requests

def ask_and_download(message, url_dest_list):
    reply = input(f"{message} (Y/N): ").strip().lower()
    if reply != 'y':
        return

    for url, dest_path in url_dest_list:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete.")
    print()

if __name__ == "__main__":

    # DeepDanio model weights
    #####################
    url_dest_list = [
        ("https://cable.ayra.ch/empty/?id=0", "deepdanio/deepdanio_data_split_0.h5"),
        ("https://cable.ayra.ch/empty/?id=0", "deepdanio/deepdanio_data_split_1.h5"),
        ("https://cable.ayra.ch/empty/?id=0", "deepdanio/deepdanio_data_split_2.h5"),
    ]
    ask_and_download("Download DeepDanio model weights?", url_dest_list)
