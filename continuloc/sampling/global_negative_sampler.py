"""Scene-wide random negatives, optionally restricted to a persisted valid grid."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class GlobalNegativeSampler:
    """Cache admissible rectangles in normalized row/column coordinates on CPU.

    NumPy's worker-local RNG is used (seeded by PyTorch DataLoader). Rectangles
    clipped at the scene sampling boundary are weighted by remaining area.
    """
    def __init__(self, sat_dataset, grid=None):
        self.scene_name = getattr(sat_dataset, 'name', 'scene')
        self.grid = grid
        self.source_hash = None
        r0,r1 = map(float, sat_dataset.nr2sample_range)
        c0,c1 = map(float, sat_dataset.nc2sample_range)
        rectangles = np.array([[r0,r1,c0,c1]],dtype=np.float64)
        if grid is not None:
            if grid.get('usage') != 'random_global_negatives_only':
                raise ValueError('Grid usage must be random_global_negatives_only')
            path=Path(grid['p_grid_npz']); meta_path=Path(grid['p_grid_metadata_json'])
            meta=json.loads(meta_path.read_text())
            self.source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as data:
                valid=data['valid_mask']; indices=data['valid_flat_indices']
                affine=data['transform'].astype(np.float64); crs=str(data['crs'].item())
            if valid.ndim!=2 or valid.dtype!=np.bool_ or indices.ndim!=1 or not np.array_equal(indices,np.flatnonzero(valid)):
                raise ValueError('Invalid grid mask/indices')
            size=float(grid['cell_size_m'])
            if (not np.isfinite(size) or size<=0 or affine.shape!=(6,) or
                not np.isfinite(affine).all() or not np.allclose(affine[[0,1,3,4]],[size,0,0,-size]) or
                not np.allclose(affine,meta['affine_transform']) or
                size!=meta['summary']['cell_size_m'] or crs!=meta['crs'] or
                valid.shape!=(meta['summary']['height'],meta['summary']['width']) or
                len(indices)!=meta['summary']['valid_cells']):
                raise ValueError('Inconsistent grid metadata')
            # Axis-aligned rectangle sampling requires the same projected CRS.
            # Reject incompatible products instead of approximating warped cells.
            if crs!=f'EPSG:{sat_dataset.epsg_code}' or sat_dataset.geo_transform[2] or sat_dataset.geo_transform[4]:
                raise ValueError('Grid requires same CRS and north-up satellite image')
            rows,cols=np.divmod(indices,valid.shape[1])
            x0=affine[0]*cols+affine[2];y0=affine[4]*rows+affine[5]
            first=sat_dataset.transfrom_georc_to_nrc(np.c_[y0,x0],source_epsg_code=sat_dataset.epsg_code,dtype=np.float64)
            last=sat_dataset.transfrom_georc_to_nrc(np.c_[y0-size,x0+size],source_epsg_code=sat_dataset.epsg_code,dtype=np.float64)
            rectangles=np.c_[np.maximum(np.minimum(first[:,0],last[:,0]),r0),
                              np.minimum(np.maximum(first[:,0],last[:,0]),r1),
                              np.maximum(np.minimum(first[:,1],last[:,1]),c0),
                              np.minimum(np.maximum(first[:,1],last[:,1]),c1)]
        if not np.isfinite(rectangles).all():raise ValueError('Nonfinite sampling bounds')
        # Keep float32 outputs strictly inside every admissible rectangle.
        rectangles[:,[0,2]]=np.nextafter(rectangles[:,[0,2]].astype(np.float32),np.float32(np.inf))
        rectangles[:,[1,3]]=np.nextafter(rectangles[:,[1,3]].astype(np.float32),np.float32(-np.inf))
        area=(rectangles[:,1]-rectangles[:,0])*(rectangles[:,3]-rectangles[:,2])
        keep=(rectangles[:,1]>rectangles[:,0])&(rectangles[:,3]>rectangles[:,2])
        self.rectangles=rectangles[keep]
        area=area[keep]
        if not len(area):raise ValueError(f'{self.scene_name}: no admissible global-negative region')
        self.probabilities=area/area.sum()

    def sample(self,num_samples,positive_nrcs=None,reject_threshold=None):
        if int(num_samples)!=num_samples or num_samples<0:raise ValueError('Negative count must be a nonnegative integer')
        target=int(num_samples)
        if not target:return torch.empty((0,2),dtype=torch.float32)
        positives=None
        if positive_nrcs is not None and reject_threshold is not None:
            positives=np.asarray(torch.as_tensor(positive_nrcs).detach().cpu(),dtype=np.float64).reshape(-1,2)
            if not np.isfinite(positives).all() or not np.isfinite(reject_threshold) or reject_threshold<0:
                raise ValueError('Invalid positive positions or rejection threshold')
        chunks=[];total=0
        for _ in range(64):
            count=min(max((target-total)*2,64),4096)
            bounds=self.rectangles[np.random.choice(len(self.rectangles),size=count,p=self.probabilities)]
            uv=np.random.random((count,2))
            candidates=np.c_[bounds[:,0]+uv[:,0]*(bounds[:,1]-bounds[:,0]),
                             bounds[:,2]+uv[:,1]*(bounds[:,3]-bounds[:,2])].astype(np.float32)
            if positives is not None and len(positives):
                conflicts=np.zeros(count,bool)
                for start in range(0,len(positives),128):
                    delta=np.abs(candidates[:,None,:]-positives[None,start:start+128,:])
                    conflicts |= (delta<=reject_threshold).all(axis=2).any(axis=1)
                candidates=candidates[~conflicts]
            chunks.append(candidates);total+=len(candidates)
            if total>=target:return torch.from_numpy(np.concatenate(chunks)[:target])
        raise RuntimeError(f'{self.scene_name}: global-negative sampling exhausted its attempt budget ({total}/{target}); no unconstrained fallback')
