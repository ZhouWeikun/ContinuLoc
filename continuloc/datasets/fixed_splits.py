"""Versioned, filename-keyed assignments for three independently split filter profiles."""
import hashlib
import json
from pathlib import Path

import numpy as np


FILTER_PROFILES = {
    'quality': (False, 'none'),
    'quality_edge': (True, 'none'),
    'quality_edge_scale': (True, 'mad'),
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_filter_profile(enable_edge_filter, scale_filter_mode):
    if not isinstance(enable_edge_filter, (bool, np.bool_)):
        raise ValueError('enable_edge_filter must be a boolean')
    mode = 'none' if scale_filter_mode is None else scale_filter_mode
    for name, flags in FILTER_PROFILES.items():
        if flags == (bool(enable_edge_filter), mode):
            return name
    raise ValueError('No fixed JSON profile for these switches: supported combinations are '
                     'edge=false/scale=none, edge=true/scale=none, edge=true/scale=mad')


def validate_fixed_metadata(path, kind, metadata_path, enable_edge_filter=True):
    if not path:
        raise ValueError('JSON splits require p_split_json')
    # Quality-only identities do not depend on which satellite map is used.
    if kind == 'satellite' and not enable_edge_filter:
        return
    source = json.loads(Path(path).read_text())['source']
    if file_sha256(metadata_path) != source[f'{kind}_metadata_sha256']:
        raise ValueError(f'{kind} metadata changed since split generation: {metadata_path}')


def read_fixed_split(frame, path, csv_path, scene_name, mode,
                     enable_edge_filter=True, scale_filter_mode='none'):
    """Select stored identities only; never run quality, edge, or scale filters."""
    profile_name = select_filter_profile(enable_edge_filter, scale_filter_mode)
    if not path or not csv_path:
        raise ValueError('JSON splits require p_split_json and p_uav_geocsv')
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 2 or data.get('scene') != scene_name:
        raise ValueError('Fixed split schema or scene does not match; use multi-profile split JSON')
    if file_sha256(csv_path) != data['source']['csv_sha256']:
        raise ValueError(f'Source CSV changed since split generation: {csv_path}')
    profile = data['profiles'][profile_name]
    if mode not in profile['protocols']:
        raise ValueError(f'Fixed split has no protocol {mode!r}')
    protocol = profile['protocols'][mode]
    names = frame[data['source']['sample_key']].astype(str).tolist()
    records = data['samples']
    by_name = {item['filename']: item for item in records}
    if len(by_name) != len(records) or len(set(names)) != len(names):
        raise ValueError('Duplicate sample filenames in CSV or fixed split')
    if set(names) != set(by_name):
        raise ValueError('CSV and fixed split must contain the same sample IDs')
    assignments = [by_name[name]['profiles'][profile_name] for name in names]
    values = [item[protocol['column']] for item in assignments]
    if any(value not in ('train', 'test', None) for value in values):
        raise ValueError('Fixed split labels must be train, test, or null')
    labels = np.array([value or '' for value in values], dtype=object)
    if not np.any(labels == 'train') or not np.any(labels == 'test'):
        raise ValueError('Fixed split must contain nonempty train and test sets')
    selected = labels != ''
    if int(selected.sum()) != profile['eligible_count']:
        raise ValueError('Fixed profile eligible_count does not match its assignments')
    if int((labels == 'train').sum()) != protocol['train_count'] or int((labels == 'test').sum()) != protocol['test_count']:
        raise ValueError('Fixed profile train/test counts do not match its assignments')
    reasons = [item.get('exclude_reason') for item in assignments]
    return dict(has_split_column=True, column_name=protocol['column'],
                split_labels=labels, selected_mask=selected,
                train_mask=labels == 'train', test_mask=labels == 'test',
                manual_filter_mask=np.array([reason == 'manual_filter' for reason in reasons]),
                train_ratio=protocol['train_ratio'], filter_profile=profile_name,
                scale_ref_m=profile['scale_ref_m'],
                diagnostics=dict(mad=profile['scale_diagnostics']['mad'],
                                 modified_z=np.array([item.get('modified_z') for item in assignments], dtype=float),
                                 mad_outlier_mask=np.array([reason == 'mad_outlier' for reason in reasons])))


def build_scene_split(scene, train_ratio):
    """Each profile filters first, then independently assigns segment/interval labels."""
    import pandas as pd
    from continuloc.datasets.uav_satellite import SatDataset, _build_split_column_name, _compute_split_indices
    csv_path = Path(scene['p_uav_geocsv'])
    before_hash = file_sha256(csv_path)
    frame = pd.read_csv(csv_path)
    key = 'filename' if 'filename' in frame else 'name'
    if frame[key].isna().any() or frame[key].astype(str).duplicated().any():
        raise ValueError(f'{scene["name"]}: missing or duplicate filenames')
    records = [dict(filename=name, profiles={}) for name in frame[key].astype(str)]
    profiles = {}
    for profile_name, (edge, scale) in FILTER_PROFILES.items():
        sat = SatDataset(
            p_satinfo_json=scene['p_satinfo_json'], p_uav_geocsv=str(csv_path),
            satellite_filepaths=scene['satellite_filepaths'], name=scene['name'],
            geo_rc_epsg_code=scene.get('geo_rc_epsg_code'),
            split_source='generated', scale_filter_mode=scale, enable_edge_filter=edge,
            load_satmaps=False, split_train_ratio=train_ratio, split_mode='segment', device='cpu')
        try:
            selection = sat.sample_selection
            eligible = np.flatnonzero(selection['selected_mask'])
            protocols = {}
            for i, record in enumerate(records):
                reason = None
                if sat.manual_filter_mask[i]:
                    reason = 'manual_filter'
                elif not selection['quality_mask'][i]:
                    reason = 'quality_rejected'
                elif edge and not selection['edge_mask'][i]:
                    reason = 'outside_satellite_sampling_range'
                elif not selection['selected_mask'][i]:
                    reason = 'mad_outlier'
                score = selection['diagnostics']['modified_z'][i]
                record['profiles'][profile_name] = dict(
                    exclude_reason=reason, modified_z=float(score) if np.isfinite(score) else None)
            for mode in ('segment', 'interval'):
                column = _build_split_column_name(mode, train_ratio)
                train, test = _compute_split_indices(len(eligible), train_ratio, mode)
                if not len(train) or not len(test):
                    raise ValueError(f'{scene["name"]}/{profile_name}: empty train/test')
                for record in records:
                    record['profiles'][profile_name][column] = None
                for label, indices in (('train', train), ('test', test)):
                    for row in eligible[indices]:
                        records[int(row)]['profiles'][profile_name][column] = label
                protocols[mode] = dict(column=column, train_ratio=train_ratio,
                                       train_count=len(train), test_count=len(test))
            profiles[profile_name] = dict(
                enable_edge_filter=edge, scale_filter_mode=scale,
                eligible_count=len(eligible), excluded_count=len(frame)-len(eligible),
                scale_ref_m=sat.scale_ref_m, edge_geometry=selection['edge_geometry'],
                scale_diagnostics=dict(mad=selection['diagnostics']['mad'],
                                       baseline_count=int(selection['edge_mask'].sum()),
                                       threshold=3.5), protocols=protocols)
        finally:
            for image in sat.satmaps:
                image.close()
    if before_hash != file_sha256(csv_path):
        raise RuntimeError(f'CSV changed during split export: {csv_path}')
    return dict(schema_version=2, scene=scene['name'], split_val_ratio=0,
                source=dict(csv_file=csv_path.name, csv_sha256=before_hash, sample_key=key,
                            satellite_metadata_file=Path(scene['p_satinfo_json']).name,
                            satellite_metadata_sha256=file_sha256(scene['p_satinfo_json']),
                            uav_metadata_file=Path(scene['p_uavinfo_json']).name,
                            uav_metadata_sha256=file_sha256(scene['p_uavinfo_json'])),
                csv_rows=len(frame), profiles=profiles, samples=records)


def export_fixed_splits(manifest_path, output_dir):
    from continuloc.config.parser import _expand_manifest_scenes
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    scenes = _expand_manifest_scenes({'dataset_manifest': str(manifest_path)})['scenes']
    if output_dir.exists():
        raise FileExistsError(f'Use a new version directory; refusing to overwrite {output_dir}')
    documents = [build_scene_split(scene, .9 if scene['dataset_root'] == 'wingta_dir' else .8)
                 for scene in scenes]
    for scene, doc in zip(scenes, documents):
        if file_sha256(scene['p_uav_geocsv']) != doc['source']['csv_sha256']:
            raise RuntimeError('CSV changed before split export completed')
    output_dir.mkdir(parents=True)
    summaries = []
    for doc in documents:
        path = output_dir/f'{doc["scene"]}.json'
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        summaries.append(dict(scene=doc['scene'], file=path.name, sha256=file_sha256(path),
                              csv_rows=doc['csv_rows'], profiles=doc['profiles']))
    index = dict(schema_version=2, dataset_name='continu_scenes',
                 manifest_sha256=file_sha256(manifest_path), scenes=summaries)
    (output_dir/'index.json').write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
    return index
