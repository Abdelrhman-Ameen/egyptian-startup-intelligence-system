"""Extract the original SAV, its dictionary, and supporting measurement definitions."""
from pathlib import Path
import json
import urllib.request
import pandas as pd
import pyreadstat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/2026-10-23/spain_technology_ventures'

def main():
    raw, processed = BASE / 'raw', BASE / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    df, meta = pyreadstat.read_sav(raw / 'technology_entrepreneurs.sav', user_missing=True)
    df.to_csv(processed / 'spain_ventures_raw_export.csv', index=False, encoding='utf-8-sig')
    attributes = ['column_names_to_labels', 'variable_value_labels', 'original_variable_types',
                  'readstat_variable_types', 'missing_ranges', 'missing_user_values',
                  'variable_measure', 'file_encoding', 'notes', 'file_label']
    info = {name: getattr(meta, name, None) for name in attributes}
    (processed / 'spss_metadata.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Shape:', df.shape, 'Encoding:', meta.file_encoding)
    print('Missing declarations:', meta.missing_ranges)
    print(df[['Age_firm', 'Employees', 'Founders_team2', 'Survival_2014']].describe().to_string())
    print('Sector examples:', repr(df.Sector.head(6).tolist()))
    for name, url in {
        'knowledge_innovation_2021': 'https://accedacris.ulpgc.es/bitstream/10553/72590/1/Preprint_doi%2010.1007-s11365-020-00656-z-2.pdf',
        'resources_strategy_2019': 'https://accedacris.ulpgc.es/bitstream/10553/41432/1/Preprint_doi%2010.1111-jsbm.12411-1.pdf',
    }.items():
        path = raw / f'{name}.pdf'
        try:
            if not path.exists():
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=40) as response:
                    path.write_bytes(response.read())
            pages = PdfReader(path).pages
            text = '\n'.join(f'\n[PAGE {i+1}]\n{p.extract_text()}' for i, p in enumerate(pages))
            (processed / f'{name}.txt').write_text(text, encoding='utf-8')
            print(f'Reference saved: {name}, {len(pages)} pages')
        except Exception as exc:
            print(f'Reference unavailable: {name}: {exc}')

if __name__ == '__main__':
    main()
