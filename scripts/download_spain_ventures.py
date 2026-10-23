"""Download the first dataset in the project handout from its original Zenodo record."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research' / '2026-10-23' / 'spain_technology_ventures'
RAW = DATA / 'raw'
RECORD_URL = 'https://zenodo.org/api/records/10607031'

def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'StartupResearch/1.0 (academic dataset analysis)'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    payload = fetch(RECORD_URL)
    metadata = json.loads(payload)
    (RAW / 'zenodo_record.json').write_bytes(payload)
    manifest = []
    for entry in metadata['files']:
        if not entry['key'].lower().endswith('.sav'):
            continue
        url = entry['links']['self']
        content = fetch(url)
        algorithm, expected = entry['checksum'].split(':', 1)
        observed = hashlib.new(algorithm, content).hexdigest()
        if observed != expected:
            raise RuntimeError('Original source checksum verification failed.')
        destination = RAW / 'technology_entrepreneurs.sav'
        destination.write_bytes(content)
        manifest.append({'source_url': url, 'original_name': entry['key'],
                         'local_path': str(destination.relative_to(ROOT)), 'bytes': len(content),
                         'source_checksum': entry['checksum'], 'checksum_verified': True,
                         'sha256': hashlib.sha256(content).hexdigest(),
                         'retrieval_date': '2026-10-23', 'record_doi': metadata['doi'],
                         'license': metadata['metadata']['license']['id']})
    if not manifest:
        raise RuntimeError('No SAV file was found in the source record.')
    (RAW / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
