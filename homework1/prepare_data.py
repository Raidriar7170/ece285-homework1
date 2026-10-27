"""Download the assigned public Kaggle archive and extract one canonical copy."""
import json
import urllib.request
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from data import file_hash

URL='https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia'


def main():
    root=Path(__file__).resolve().parent/'data'; root.mkdir(exist_ok=True)
    archive=root/'chest-xray.zip'
    if not archive.exists():
        partial=root/'chest-xray.zip.part'
        urllib.request.urlretrieve(URL, partial)
        partial.rename(archive)
    count=0
    with ZipFile(archive) as z:
        for info in z.infolist():
            parts=PurePosixPath(info.filename).parts
            # Archive also contains a duplicated nested tree and Mac metadata.
            if len(parts)!=4 or parts[0]!='chest_xray' or parts[1] not in {'train','val','test'}:
                continue
            if parts[2] not in {'NORMAL','PNEUMONIA'} or Path(parts[3]).suffix.lower() not in {'.jpeg','.jpg','.png'}:
                continue
            out=root.joinpath(*parts)
            out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():
                out.write_bytes(z.read(info))
            count+=1
    (root/'download_manifest.json').write_text(json.dumps({'url':URL,'archive_sha256':file_hash(archive),
         'extracted_images':count,'policy':'canonical chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}; nested duplicate and metadata excluded'},indent=2))
    print(f'Prepared {count} images at {root / "chest_xray"}',flush=True)


if __name__=='__main__':
    main()
