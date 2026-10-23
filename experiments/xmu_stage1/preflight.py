"""Read-only audit of the legacy XMU dataset against the current loader contract."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed')
META = DATA / 'dataset_xmu_meta'

def main():
    csv_path = META / 'uavimgs_geoloc_xiangan_xmu_dji.csv'
    with csv_path.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    required = {'filename', 'latitude', 'longitude', 'rotdeg_fm_north_anticlock',
                'h_cover_m', 'aff2d_corrected'}
    sat = json.loads((META / 'satimg_xiangan_xmu5km2_03res_info.json').read_text())
    uav = json.loads((META / 'uavimgs_xiangan_xmu_info.json').read_text())
    missing = [str(DATA / r['Line'] / r['Name']) for r in rows
               if not (DATA / r['Line'] / r['Name']).is_file()]
    report = {
        'source_csv': str(csv_path), 'rows': len(rows),
        'flight_lines': sorted({r['Line'] for r in rows}),
        'missing_image_count': len(missing), 'missing_image_examples': missing[:10],
        'satellite_exists': Path(sat['tif_path']).is_file(),
        'missing_csv_fields': sorted(required - set(fields)),
        'missing_satellite_metadata': sorted({'epsg_code', 'filepaths', 'means_normalized',
                                              'stds_normalized'} - set(sat)),
        'missing_uav_metadata': sorted({'geo_rc_epsg_code'} - set(uav)),
        'status': 'blocked_pending_coverage_and_registration_metadata',
        'notes': ['Name/Latitude/Longitude can be renamed; filenames need Line/Name paths.',
                  'Projected coordinate columns must match geo_rc_epsg_code.',
                  'Do not invent aff2d_corrected or substitute RltAlt for h_cover_m.',
                  'Original data is unchanged; no training has been launched.'],
    }
    output = ROOT / 'gen_fm_exps/xmu_stage1/preflight.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
