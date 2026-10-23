"""Offline audit of saved predictions; no model inference or checkpoint changes."""
import argparse
import csv
import html
import json
import sys
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from master_pose_arrows import _draw_pose_marker, _nrc_to_crop_xy, _wrap_pi

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Figure configuration (including all rendered labels).
TILE = 256
CANVAS_SIZE = (1280, 415)
FONT_SIZE = 14
TITLE = "Query {qid:03d} | {filename} | red: GT, blue: prediction (arrow length encodes scale)"
LABELS = ["Query (center square)", "GT / satellite {index}", "Coarse top1", "Mode top1", "Final top1"]
ERROR = "{dist:.1f} m | rot {rot:.1f} deg\npred/GT {signed:.3f} | sym {ratio:.3f}\ncoverage {cover:.1f} m\npred/scene median {to_median:.3f}\npred/scene mean {to_mean:.3f}"
GT_TEXT = "GT coverage {cover:.1f} m\nGT/scene median {to_median:.3f}\nGT/scene mean {to_mean:.3f}"
SCENE_TEXT = "Scene scale population: {count} selected rows (before spatial edge filtering and train/test split). Coverage median={median:.3f} m, mean={mean:.3f} m; normalized scale median={median_scale:.4f}, mean={mean_scale:.4f}; normalization reference={ref:.1f} m. Reconstructed from the current CSV using the saved run selection settings."
FIGSIZE = (11, 4)
PLOT_LABELS = ["GT coverage (m)", "Predicted coverage (m)", "Prediction / GT", "Queries"]
PLOT_TITLE = ["Scale estimate: coarse vs final", "Final signed scale bias"]
LEGENDS = ["Coarse", "Final", "Ideal"]
GALLERY_TITLE = "{scene} {split} | scale={scales}, particles={particles}"
GALLERY_HELP = "All {count} saved queries. Red=GT, blue=prediction. Sort/filter by final errors. Satellite backdrop: {satellite}. Coordinates and metrics come from saved predictions; images are offline diagnostic crops."
GALLERY_SAMPLE_TITLE = (
    "Query {qid:03d} | {filename} | CSV row index (0-based): {source_csv_row}\n"
    "Final: position {dist:.2f} m | rot {rot:.2f} deg | pred/GT {signed:.4f} | sym {ratio:.4f}\n"
    "Coverage: GT {gt_coverage_m:.2f} m | prediction {pred_coverage_m:.2f} m | category: {category}\n"
    "Scene median: {scene_median_coverage_m:.3f} m | GT/median {gt_to_scene_median:.4f} | pred/median {pred_to_scene_median:.4f}\n"
    "Scene mean: {scene_mean_coverage_m:.3f} m | GT/mean {gt_to_scene_mean:.4f} | pred/mean {pred_to_scene_mean:.4f}"
)
SORT_LABELS = {"qid": "Query order", "dist": "Position error", "rot": "Rotation error", "ratio": "Scale error"}
FILTER_LABELS = {"all": "All", "scale": "Position + rotation pass, scale fails", "position": "Position fails", "rotation": "Position passes, rotation fails", "success": "All pass"}


def project_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "--p_yaml", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--run", type=Path, help="Override the configured Stage 3 result directory")
    ap.add_argument("--output", type=Path, help="Override the configured output directory")
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    unknown = set(config) - {"run", "output", "satellite_index", "font_path"}
    if unknown:
        raise ValueError(f"Unknown analysis configuration keys: {sorted(unknown)}")
    args.run = project_path(args.run or config["run"])
    out = project_path(args.output or config["output"])
    if not out.is_relative_to(ROOT / "gen_fm_exps") or out == args.run:
        raise ValueError("Output must be a separate analysis directory under gen_fm_exps/")
    satellite_index = int(config.get("satellite_index", 0))
    run_config = yaml.safe_load((args.run / "config.yaml").read_text())
    scene_name = run_config["exp_setting"]["selected_scene_name"]
    scenes = run_config["scenes_setting"]["scenes"]
    scene = next(s for s in scenes if s["name"] == scene_name)
    stage2_config = yaml.safe_load((args.run / "stage2_opts.yaml").read_text())
    inference = run_config["inference_setting"]
    gallery_title = GALLERY_TITLE.format(
        scene=scene_name, split=stage2_config["data_setting"]["split_mode"],
        scales=inference["hds_stage1_n_scale"], particles=inference["hds_stage2_particles_m"])
    p = torch.load(args.run / "predictions.pt", map_location="cpu", weights_only=True)
    p = {k: v.numpy() for k, v in p.items() if torch.is_tensor(v)}
    gt = p["coords_gt"]
    report = json.loads((args.run / "stage3_progressive_recall_report.json").read_text())
    factor = report["thresholds"]["nrc2meter"]
    meta = json.loads(project_path(scene["p_satinfo_json"]).read_text())
    uav = json.loads(project_path(scene["p_uavinfo_json"]).read_text())
    if not 0 <= satellite_index < len(meta["filepaths"]):
        raise ValueError("satellite_index is outside the available satellite images")
    satellite_path = project_path(meta["filepaths"][satellite_index])
    gallery_help = GALLERY_HELP.format(count=len(gt), satellite=satellite_path.name)
    with project_path(scene["p_uav_geocsv"]).open() as f:
        rows = list(csv.DictReader(f))
    size = max(meta["img_h_px"], meta["img_w_px"])
    geo = meta["geo_transform"]
    epsg = int(uav["geo_rc_epsg_code"])
    if epsg != int(meta["epsg_code"]) or geo[2] != 0 or geo[4] != 0:
        raise ValueError("This exporter requires matching projected CRS and an unrotated satellite raster")
    rc = np.array([[(float(r[f"geo_row_proj{epsg}"]) - geo[3]) / geo[5] / size,
                    (float(r[f"geo_col_proj{epsg}"]) - geo[0]) / geo[1] / size] for r in rows])
    matches = np.argmin(((gt[:, None, :2] - rc[None]) ** 2).sum(2), axis=1)
    assert len(set(matches)) == len(gt), "Ambiguous image mapping"
    assert np.max(np.abs(gt[:, :2] - rc[matches])) < 1e-6, "GT position mismatch"
    refs = np.array([float(rows[i]["h_cover_m"]) for i in matches]) / gt[:, 3]
    ref = round(float(np.median(refs)), 3)
    assert np.allclose(refs, ref, atol=1e-4), "Scale reference mismatch"
    # Reuse the same scene population that defines SatDataset scale statistics.
    import pandas as pd
    from continuloc.datasets.dataset_neuloc_4d import _resolve_sample_selection
    data_settings = {**stage2_config.get("data_setting", {}),
                     **run_config.get("data_setting", {})}
    selection = _resolve_sample_selection(
        pd.DataFrame(rows), data_settings.get("split_mode", "segment"),
        data_settings.get("split_train_ratio", 0.9),
        (abs(meta["x_resolution_m"]) + abs(meta["y_resolution_m"])) / 2,
        run_config["scenes_setting"].get("dataset_name", scene_name),
        scale_ref_m=ref, enable_scale_filter=data_settings.get("enable_scale_filter", False),
        scale_filter_mode=data_settings.get("scale_filter_mode"),
        split_source=data_settings.get("split_source", "generated"))
    selected_indices = np.flatnonzero(selection["selected_mask"])
    assert np.all(selection["selected_mask"][matches]), "Saved queries no longer pass scene selection"
    scene_cover = np.array([float(rows[i]["h_cover_m"]) for i in selected_indices])
    scene_median, scene_mean = float(np.median(scene_cover)), float(np.mean(scene_cover))
    scene_stats = dict(count=len(scene_cover), median=scene_median, mean=scene_mean,
                       median_scale=scene_median/ref, mean_scale=scene_mean/ref, ref=ref)
    scene_text = SCENE_TEXT.format(**scene_stats)
    scene_stats.update(population="selected scene rows before spatial edge filtering and train/test split",
                       csv_path=str(project_path(scene["p_uav_geocsv"])),
                       csv_sha256=hashlib.sha256(project_path(scene["p_uav_geocsv"]).read_bytes()).hexdigest(),
                       selection_settings=data_settings)
    csv_rot = np.deg2rad([float(rows[i]["rotdeg_fm_north_anticlock"]) for i in matches])
    assert np.max(np.abs((csv_rot - gt[:, 2] + np.pi) % (2*np.pi) - np.pi)) < 1e-5
    preds = {}
    metrics = {}
    for stage in ["grid", "mode", "evo"]:
        # Explicitly rank using saved scores, as the evaluator does.
        ix = np.argmax(p["scores_" + stage], axis=1)
        pred = p["coords_" + stage][np.arange(len(gt)), ix]
        preds[stage] = pred
        signed = pred[:, 3] / gt[:, 3]
        metrics[stage] = dict(
            dist=np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1) * factor,
            rot=np.abs((pred[:, 2] - gt[:, 2] + np.pi) % (2*np.pi) - np.pi) * 180/np.pi,
            signed=signed, ratio=np.maximum(signed, 1/signed))
    m = metrics["evo"]
    thresholds = report["thresholds"]
    pos = m["dist"] <= thresholds["dist_th_meter"]
    rot = m["rot"] <= thresholds["rot_th"]
    scale = m["ratio"] <= thresholds["scale_ratio_th"]
    counts = [int(pos.sum()), int((pos & rot).sum()), int((pos & rot & scale).sum())]
    expected = [report["stages"]["seed_mode_final"]["progressive_error_metrics"][k]["n_success_top1"]
                for k in ["dist_recall", "dist_rot_recall", "dist_rot_scale_recall"]]
    assert counts == expected, (counts, expected)
    (out / "images").mkdir(parents=True, exist_ok=True)
    config.update(run=str(args.run), output=str(out), satellite_index=satellite_index)
    (out / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    summary = dict(run=str(args.run.resolve()), scale_ref_m=ref, n=len(gt),
                   scene_scale_statistics=scene_stats,
                   success_counts=counts, scale_only_fail=int((pos & rot & ~scale).sum()),
                   stages={s: dict(coverage_percentiles=np.percentile(v[:, 3]*ref, [0,5,50,95,100]).tolist(),
                                   signed_ratio_percentiles=np.percentile(metrics[s]["signed"], [0,5,50,95,100]).tolist())
                           for s,v in preds.items()},
                   gt_coverage_percentiles=np.percentile(gt[:, 3]*ref, [0,5,50,95,100]).tolist())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    records = []
    for q in range(len(gt)):
        rec = dict(qid=q, source_csv_row=int(matches[q]), filename=rows[matches[q]]["filename"],
                   gt_coverage_m=float(gt[q,3]*ref), pred_coverage_m=float(preds["evo"][q,3]*ref),
                   scene_median_coverage_m=scene_median, scene_mean_coverage_m=scene_mean,
                   gt_to_scene_median=float(gt[q,3]*ref/scene_median),
                   gt_to_scene_mean=float(gt[q,3]*ref/scene_mean),
                   pred_to_scene_median=float(preds["evo"][q,3]*ref/scene_median),
                   pred_to_scene_mean=float(preds["evo"][q,3]*ref/scene_mean),
                   **{k: float(v[q]) for k,v in m.items()},
                   category="success" if pos[q] and rot[q] and scale[q] else
                            "scale" if pos[q] and rot[q] else "position" if not pos[q] else "rotation")
        for s in metrics:
            rec.update({s+"_"+k: float(v[q]) for k,v in metrics[s].items()})
            rec.update({s+"_to_scene_median": float(preds[s][q,3]*ref/scene_median),
                        s+"_to_scene_mean": float(preds[s][q,3]*ref/scene_mean)})
        records.append(rec)
    with (out / "per_query.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(1,2,figsize=FIGSIZE)
    for s,label in zip(["grid","evo"],LEGENDS):
        ax[0].scatter(gt[:,3]*ref,preds[s][:,3]*ref,s=8,alpha=.6,label=label)
    lim = [float(gt[:,3].min()*ref),float(gt[:,3].max()*ref)]
    ax[0].plot(lim,lim,"k--",label=LEGENDS[2])
    ax[0].set(xlabel=PLOT_LABELS[0],ylabel=PLOT_LABELS[1],title=PLOT_TITLE[0])
    ax[0].legend()
    ax[1].hist(m["signed"],bins=35)
    for x in [1/thresholds["scale_ratio_th"],1,thresholds["scale_ratio_th"]]:
        ax[1].axvline(x,color="red" if x!=1 else "black",linestyle="--")
    ax[1].set(xlabel=PLOT_LABELS[2],ylabel=PLOT_LABELS[3],title=PLOT_TITLE[1])
    fig.tight_layout()
    fig.savefig(out / "scale_diagnostics.png",dpi=160)
    plt.close(fig)
    Image.MAX_IMAGE_PIXELS = None
    sat = np.array(Image.open(satellite_path).convert("RGB"))
    res = (abs(meta["x_resolution_m"]) + abs(meta["y_resolution_m"])) / 2
    font = ImageFont.truetype(str(project_path(config["font_path"])), FONT_SIZE)
    xx, yy = np.meshgrid(np.linspace(-1,1,TILE),np.linspace(-1,1,TILE))

    def crop(c):
        half = c[3] * ref / res / 2
        x,y = xx*half,yy*half
        col = c[1]*size + np.cos(c[2])*x + np.sin(c[2])*y
        row = c[0]*size - np.sin(c[2])*x + np.cos(c[2])*y
        return Image.fromarray(cv2.remap(sat,col.astype("float32"),row.astype("float32"),cv2.INTER_LINEAR))

    for q,rec in enumerate(records):
        query = Image.open(project_path(uav["uavimgs_dir"]) / rec["filename"]).convert("RGB")
        w,h = query.size
        edge = min(w,h)
        query = query.crop(((w-edge)//2,(h-edge)//2,(w+edge)//2,(h+edge)//2)).resize((TILE,TILE))
        images = [query,crop(gt[q])]
        for s in ["grid","mode","evo"]:
            c = preds[s][q]
            im = crop(c)
            draw = ImageDraw.Draw(im)
            _draw_pose_marker(draw,center=((TILE-1)/2,)*2,angle_rad=0,length_px=40,color=(30,144,255))
            xy = _nrc_to_crop_xy(gt_nr=gt[q,0],gt_nc=gt[q,1],pred_nr=c[0],pred_nc=c[1],
                                pred_rot=c[2],pred_scale=c[3],halfimg_radius_nrc=ref/res/size/2,width=TILE,height=TILE)
            _draw_pose_marker(draw,center=xy,angle_rad=_wrap_pi(float(gt[q,2]-c[2])),
                              length_px=float(40*gt[q,3]/c[3]),color=(255,30,30))
            images.append(im)
        canvas = Image.new("RGB",CANVAS_SIZE,"white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5,3),TITLE.format(qid=q,filename=rec["filename"]),font=font,fill="black")
        for j,im in enumerate(images):
            canvas.paste(im,(j*TILE,45))
            draw.text((j*TILE+4,25),LABELS[j].format(index=satellite_index),font=font,fill="black")
            text = ""
            if j==1:
                text = GT_TEXT.format(cover=gt[q,3]*ref, to_median=rec["gt_to_scene_median"],
                                      to_mean=rec["gt_to_scene_mean"])
            elif j>=2:
                s = ["grid","mode","evo"][j-2]
                text = ERROR.format(**{k:v[q] for k,v in metrics[s].items()},cover=preds[s][q,3]*ref,
                                    to_median=rec[s+"_to_scene_median"], to_mean=rec[s+"_to_scene_mean"])
            draw.text((j*TILE+4,305),text,font=font,fill="black")
        canvas.save(out / "images" / f"{q:03d}.jpg",quality=90)
        if q % 75 == 0:
            print("Rendered",q,flush=True)
    options = lambda d: "".join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in d.items())
    gallery_records = [dict(r, display_title=GALLERY_SAMPLE_TITLE.format(**r)) for r in records]
    records_json = json.dumps(gallery_records).replace("<", "\\u003c")
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(gallery_title)}</title>
<style>body{{font:16px sans-serif;background:#eee;margin:20px}}img{{width:100%;max-width:1280px}}article{{margin:20px 0}}h2{{font:inherit;font-weight:600;white-space:pre-wrap;overflow-wrap:anywhere;user-select:text;background:white;padding:12px;max-width:1256px}}header{{position:sticky;top:0;background:white;padding:12px}}select{{padding:8px}}</style>
<h1>{html.escape(gallery_title)}</h1><p>{html.escape(gallery_help)}</p><p>{html.escape(scene_text)}</p><img src="scale_diagnostics.png">
<header><select id="sort">{options(SORT_LABELS)}</select><select id="filter">{options(FILTER_LABELS)}</select><span id="count"></span></header><main></main>
<script>const rows={records_json};function render(){{let key=document.querySelector('#sort').value,filter=document.querySelector('#filter').value;
let selected=rows.filter(r=>filter==='all'||r.category===filter).sort((a,b)=>key==='qid'?a.qid-b.qid:b[key]-a[key]);
document.querySelector('#count').textContent=' '+selected.length+' queries';
const main=document.querySelector('main');main.replaceChildren();
selected.forEach(r=>{{const article=document.createElement('article'),title=document.createElement('h2'),img=document.createElement('img');
title.textContent=r.display_title;img.loading='lazy';img.src=`images/${{String(r.qid).padStart(3,'0')}}.jpg`;
img.alt=r.display_title;article.append(title,img);main.append(article);}});}}
document.querySelectorAll('select').forEach(s=>s.onchange=render);render();</script>"""
    (out / "index.html").write_text(page)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
