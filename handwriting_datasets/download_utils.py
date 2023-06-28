import gdown
import hashlib
import pathlib as pl
import subprocess

def resumable_download(url, root: str, filename: str, google=False) -> None:
    """
    TODO: factor out in utility module

    Args:
        url: URL to download (may not contain a filename)
        root: saving directory
        filename: name to use for saving the file
        google: URL is a Google drive shared link
    """
    if not pl.Path( root ).exists() or not pl.Path( root ).is_dir():
        print(f'Saving path {root} does not exist! Download for {url} aborted.')
        return
    ## for downloading large from Google drive
    if google:
        gdown.download( url, str(pl.Path(root, filename)), resume=True )
    else:
        print(f"Downloading {url} ({filename}) ...")
        cmd = 'wget --directory-prefix={} -c {}'.format( root, url )
        subprocess.run( cmd, shell=True, check=True)
    print(f"Done with {root}/{filename}")


def is_valid_archive(file_path: pl.Path, md5: str) -> bool:
    if not file_path.exists():
        return False
    return hashlib.md5( open(file_path, 'rb').read()).hexdigest() == md5


