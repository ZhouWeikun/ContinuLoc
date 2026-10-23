"""Frozen Stage1 retrieval on legacy XMU metadata, with explicit metric gallery size."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def rank_rotation_candidates(responses, warp_angles, top_k, mode, temperature):
    """Return xy indices, discrete headings and the actual ranking scores."""
    import torch
    from continuloc.evaluation.query_rotation_search import (
        exponential_location_scores, rotation_search_ranking,
    )
    if mode in ('rot_weighting_argmax', 'rot_sum_argmax'):
        # Aggregate locations, but retain a discrete maximum-response heading.
        location_scores = (responses.sum(dim=1) if mode == 'rot_sum_argmax'
                           else exponential_location_scores(responses, temperature))
        score, idx = location_scores.topk(min(top_k, responses.shape[2]), dim=1)
        angle_idx = responses.argmax(dim=1).gather(1, idx)
        angles = warp_angles.to(device=responses.device, dtype=responses.dtype)
        warp = angles[angle_idx]
        heading = torch.atan2(torch.sin(-warp), torch.cos(-warp))
    elif mode in ('rot_argmax', 'rot_xy_argmax'):
        idx, heading = rotation_search_ranking(
            responses, warp_angles, top_k, mode=mode, return_all_headings=True)
        if mode == 'rot_argmax':
            score = responses.max(dim=1).values.gather(1, idx)
        else:
            score = responses.flatten(1).topk(min(top_k, responses.shape[1] * responses.shape[2]), dim=1).values
    else:
        raise ValueError(f'Unknown ranking mode: {mode}')
    return idx, heading, score

def main():
    import yaml
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    config_args, _ = config_parser.parse_known_args()
    config = yaml.safe_load(config_args.config.read_text())
    if not isinstance(config, dict):
        config_parser.error('Config must be a mapping')
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--crop-m', type=float, default=100.)
    parser.add_argument('--stride-m', type=float, default=25.)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-queries', type=int, default=0)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--rotation-mode', choices=['metadata', 'search', 'none'], default='metadata')
    parser.add_argument('--rot-retrieval-mode', choices=['rot_weighting_argmax', 'rot_sum_argmax', 'rot_xy_argmax', 'rot_argmax'], default='rot_weighting_argmax')
    parser.add_argument('--heading-temperature', type=float, default=0.02)
    parser.add_argument('--rotation-step-deg', type=float, default=10.)
    parser.add_argument('--max-forward-variants', type=int, default=64)
    unknown = set(config) - {action.dest for action in parser._actions}
    if unknown:
        parser.error(f'Unknown config keys: {sorted(unknown)}')
    parser.set_defaults(**config)
    args = parser.parse_args()
    for key in ('checkpoint', 'data_root', 'output_root'):
        value = getattr(args, key)
        if value is None:
            parser.error(f'Missing {key}')
        path = Path(value)
        setattr(args, key, path if path.is_absolute() else ROOT / path)
    if args.crop_m <= 0 or args.stride_m <= 0 or args.batch_size < 1 or args.max_queries < 0:
        parser.error('Invalid crop, stride, batch size or query limit')
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from pyproj import Geod
    from torchvision import transforms as T
    from continuloc.config.parser import get_parse
    from continuloc.engine.stage1_visual_encoder_minimal import MinimalStage1VisualEncoderTrainer
    from continuloc.evaluation.query_rotation_search import rotation_steps_rad, resolve_heading_temperature
    if args.max_forward_variants < 1:
        parser.error('max_forward_variants must be positive')
    if args.rotation_mode == 'search' and args.rot_retrieval_mode == 'rot_weighting_argmax':
        resolve_heading_temperature(args.heading_temperature)
    warp_angles = rotation_steps_rad(args.rotation_step_deg) if args.rotation_mode == 'search' else None
    torch.set_num_threads(4)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.argv = [sys.argv[0], '--p_yaml', str(checkpoint.parent / 'opts.yaml')]
    opt = get_parse()
    opt.load2test = str(checkpoint)
    opt.gpu_ids = [str(args.gpu_id)]
    trainer = MinimalStage1VisualEncoderTrainer(opt)
    # Strict loading prevents accidentally evaluating a partly initialized model.
    state = torch.load(checkpoint, map_location='cpu')
    for key in ('vis_encoder', 'vis_aggregator'):
        getattr(trainer, key).load_state_dict(state[key], strict=True)
        getattr(trainer, key).eval().requires_grad_(False)
    data = args.data_root
    meta = data / 'dataset_xmu_meta'
    satinfo = json.loads((meta / 'satimg_xiangan_xmu5km2_03res_info.json').read_text())
    uavinfo = json.loads((meta / 'uavimgs_xiangan_xmu_info.json').read_text())
    with (meta / 'uavimgs_geoloc_xiangan_xmu_dji.csv').open() as f:
        rows = list(csv.DictReader(f))
    if args.max_queries:
        rows = rows[:args.max_queries]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output_root / stamp
    out.mkdir(parents=True, exist_ok=False)
    cfg = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           'checkpoint': str(checkpoint), 'query_count': len(rows),
           'heading': 'discrete full-circle search without metadata alignment' if args.rotation_mode == 'search' else ('metadata north alignment' if args.rotation_mode == 'metadata' else 'original orientation; no rotation'),
           'ranking': args.rot_retrieval_mode if args.rotation_mode == 'search' else 'cosine_xy',
           'score_type': 'logsumexp(-chord_distance/temperature)' if args.rotation_mode == 'search' and args.rot_retrieval_mode == 'rot_weighting_argmax' else ('sum_cosine_over_rot' if args.rotation_mode == 'search' and args.rot_retrieval_mode == 'rot_sum_argmax' else 'cosine'),
           'angle_count': len(warp_angles) if warp_angles is not None else 1, 'scale': 'fixed gallery size; query scale unchanged',
           'metric': 'WGS84 geodesic center distance', 'partition': 'all queries, no training split',
           'satellite': satinfo['tif_path'], 'checkpoint_size': checkpoint.stat().st_size}
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg))
    (out / 'model_options.yaml').write_text(yaml.safe_dump(vars(opt)))
    print(f'OUTPUT_DIR={out}', flush=True)
    sat = Image.open(satinfo['tif_path'])
    assert 4326 in sat.tag_v2[34735], 'Expected geographic WGS84 source raster'
    sat.load()
    gt = satinfo['geo_transform']
    assert gt[2] == 0 and gt[4] == 0 and gt[1] > 0 and gt[5] < 0
    rx, ry = satinfo['x_resolution_m'], satinfo['y_resolution_m']
    cw, ch = args.crop_m / rx, args.crop_m / ry
    # Pixel coordinates refer to image edges; complete crops only, no artificial padding.
    xs = np.arange(cw / 2, sat.width - cw / 2 + 1e-6, args.stride_m / rx)
    ys = np.arange(ch / 2, sat.height - ch / 2 + 1e-6, args.stride_m / ry)
    centers = np.array([(x, y) for y in ys for x in xs])
    assert len(centers) >= 10
    gallery_ll = np.column_stack((gt[0] + centers[:, 0] * gt[1], gt[3] + centers[:, 1] * gt[5]))
    size = int(opt.imgsize2net)
    def transform(info):
        return T.Compose([T.ToTensor(), T.Normalize(info['mean'], info['std'])])
    st, ut = transform(satinfo), transform(uavinfo)
    feats = []
    gallery_batch = min(args.batch_size, args.max_forward_variants)
    query_batch = min(args.batch_size, 64)  # Bound rotation response matrix memory.
    with torch.inference_mode():
        for start in range(0, len(centers), gallery_batch):
            images = [st(sat.resize((size, size), Image.Resampling.BICUBIC,
                        box=(x-cw/2, y-ch/2, x+cw/2, y+ch/2)))
                      for x, y in centers[start:start+gallery_batch]]
            f = trainer._encode_imgs_eval(torch.stack(images).to(trainer.device))
            assert torch.isfinite(f).all()
            feats.append(f.cpu())
            if start % (gallery_batch * 10) == 0:
                print(f'Gallery {start}/{len(centers)}', flush=True)
        gallery = torch.cat(feats)
        torch.save({'features': gallery, 'lonlat': gallery_ll, 'pixel_centers': centers}, out / 'gallery.pt')
        gallery_gpu = gallery.to(trainer.device)
        query_feats, indices, scores, headings = [], [], [], []
        for start in range(0, len(rows), query_batch):
            batch = rows[start:start+query_batch]
            images = []
            for r in batch:
                with Image.open(data / r['Line'] / r['Name']) as im:
                    im = im.convert('RGB')
                    w, h = im.size
                    side = min(w, h)
                    im = im.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
                    images.append(ut(im.resize((size, size), Image.Resampling.BICUBIC)))
            imgs = torch.stack(images).to(trainer.device)
            if args.rotation_mode == 'search':
                responses = trainer._rotation_search_responses(
                    imgs, torch.ones(len(batch), device=trainer.device),
                    warp_angles, gallery_gpu, args.max_forward_variants)
                assert torch.isfinite(responses).all()
                idx, heading, score = rank_rotation_candidates(
                    responses, warp_angles, 10, args.rot_retrieval_mode, args.heading_temperature)
                headings.append(heading.cpu())
                del responses
            else:
                if args.rotation_mode == 'metadata':
                    rot = torch.tensor([-np.deg2rad(float(r['rotdeg_fm_north_anticlock'])) for r in batch],
                                       device=trainer.device, dtype=imgs.dtype)
                    imgs = trainer._warp_uav_imgs(imgs, rot_rad=rot)
                f = trainer._encode_imgs_eval(imgs)
                assert torch.isfinite(f).all()
                score, idx = (f @ gallery_gpu.T).topk(10, dim=1)
                query_feats.append(f.cpu())
            indices.append(idx.cpu()); scores.append(score.cpu())
            if start % (query_batch * 10) == 0:
                print(f'Queries {start}/{len(rows)}', flush=True)
    idx = torch.cat(indices).numpy()
    qll = np.array([[float(r['Longitude']), float(r['Latitude'])] for r in rows])
    predicted = gallery_ll[idx]
    geod = Geod(ellps='WGS84')
    distances = geod.inv(np.broadcast_to(qll[:, 0, None], idx.shape),
                         np.broadcast_to(qll[:, 1, None], idx.shape),
                         predicted[:, :, 0], predicted[:, :, 1])[2]
    metrics = {'queries': len(rows), 'gallery': len(centers),
               'top1_median_m': float(np.median(distances[:, 0])),
               'top1_mean_m': float(np.mean(distances[:, 0])),
               'recall_percent': {str(m): {str(k): float(100 * (distances[:, :k].min(1) <= m).mean())
                                          for k in (1, 5, 10)} for m in (25, 50, 100)}}
    extra = {}
    if headings:
        heading_deg = np.rad2deg(torch.cat(headings).numpy())
        gt_heading_deg = np.array([float(r['rotdeg_fm_north_anticlock']) for r in rows])
        error = np.abs((heading_deg[:, 0] - gt_heading_deg + 180) % 360 - 180)
        metrics['top1_heading_error_mean_deg'] = float(error.mean())
        metrics['top1_heading_error_median_deg'] = float(np.median(error))
        extra = dict(top10_heading_deg=heading_deg, gt_heading_deg=gt_heading_deg,
                     top1_heading_error_deg=error, warp_angles_deg=np.rad2deg(warp_angles.numpy()))
    np.savez_compressed(out / 'retrieval.npz', **extra, query_lonlat=qll, gallery_lonlat=gallery_ll,
                        top10_indices=idx, top10_scores=torch.cat(scores).numpy(), distances_m=distances,
                        filenames=np.array([r['Line']+'/'+r['Name'] for r in rows]))
    if query_feats:
        torch.save(torch.cat(query_feats), out / 'query_features.pt')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'completed.json').write_text(json.dumps({'status': 'complete'}))
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
