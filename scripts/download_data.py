"""Download only the official Kaggle Base file, anonymously, and record provenance."""
from pathlib import Path
import sys, json, hashlib, zipfile, urllib.request
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[1]
URL='https://www.kaggle.com/api/v1/datasets/download/sgpjesus/bank-account-fraud-dataset-neurips-2022/Base.csv'
EXPECTED_BYTES=213427735

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def main():
    dest=ROOT/'data/raw'; dest.mkdir(parents=True,exist_ok=True)
    csv=dest/'Base.csv'; archive=dest/'Base.csv.download'
    if not csv.exists():
        print('Downloading official BAF Base only...',flush=True)
        urllib.request.urlretrieve(URL,archive)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                assert 'Base.csv' in z.namelist()
                with z.open('Base.csv') as source,open(csv,'wb') as out:
                    import shutil
                    shutil.copyfileobj(source,out)
            archive.unlink()  # keep only the extracted CSV
        else:
            archive.rename(csv)
    assert csv.stat().st_size==EXPECTED_BYTES, 'Unexpected file size; inspect upstream version before using.'
    metadata={'dataset':'Bank Account Fraud Dataset Suite, Base','official_owner':'sgpjesus',
      'official_repository':'https://github.com/feedzai/bank-account-fraud',
      'dataset_page':'https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022',
      'download_url':URL,'kaggle_version':2,'filename':'Base.csv','bytes':csv.stat().st_size,
      'sha256':sha(csv),'verified_at_singapore':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'note':'Public synthetic benchmark derived from anonymized real bank applications; unmodified original CSV retained.',
      'license_note':'See the upstream Kaggle license and original references/LICENSE; datasheet and Kaggle contain differing license text. No redistribution or deployment license is inferred here.'}
    (dest/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
