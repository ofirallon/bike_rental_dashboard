"""Download the original UCI hourly file; never extract arbitrary archive paths."""
import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
ROOT=Path(__file__).resolve().parents[1]


def download(output):
    if output.exists():
        raise FileExistsError(f'{output} already exists; keep it or move it before downloading')
    with urlopen(URL,timeout=60) as response:
        archive=response.read()
    with ZipFile(BytesIO(archive)) as zipped:
        matches=[name for name in zipped.namelist() if Path(name).name=='hour.csv']
        if len(matches)!=1:
            raise ValueError('Archive must contain exactly one hour.csv')
        content=zipped.read(matches[0])
    if b'dteday' not in content.splitlines()[0] or b'cnt' not in content.splitlines()[0]:
        raise ValueError('Unexpected hourly file schema')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(content)
    print(f'Downloaded original UCI hour.csv to {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'hour.csv')
    download(parser.parse_args().output)
