"""Partition metrics after a single pass over held-out query/gallery matches."""
import numpy as np
import torch


class PartitionedRecall:
    def __init__(self, labels, k_values):
        self.labels = np.asarray(labels, dtype=object)
        if not np.isin(self.labels, ['val', 'test']).all():
            raise ValueError('Evaluation labels must be val/test')
        self.k_values = list(k_values)
        self.offset = 0
        self.groups = {name: {'count': 0, 'hits': {k: 0 for k in k_values}, 'distances': []}
                       for name in ('val', 'test') if np.any(self.labels == name)}

    def update(self, hits, top1_distances):
        hits = hits.detach().cpu()
        distances = top1_distances.detach().cpu()
        n = len(distances)
        if hits.shape[0] != n or self.offset + n > len(self.labels):
            raise ValueError('Evaluation outputs do not match query membership/order')
        labels = self.labels[self.offset:self.offset+n]
        self.offset += n
        for name, group in self.groups.items():
            selected = torch.from_numpy(labels == name)
            group['count'] += int(selected.sum())
            for k in self.k_values:
                # Recall@K uses all available candidates when the gallery is smaller than K.
                group['hits'][k] += int(hits[selected, :k].any(dim=1).sum())
            if selected.any():
                group['distances'].append(distances[selected])

    def finish(self, nrc2meter, label, logger=None):
        if self.offset != len(self.labels):
            raise ValueError('Not all evaluation queries were processed')
        output = {}
        for name, group in self.groups.items():
            n = group['count']
            result = {f'recall@{k}': group['hits'][k] / n for k in self.k_values}
            result['num_queries'] = n
            distances = torch.cat(group['distances']) * nrc2meter
            result['top1_dist_meter_mean'] = float(distances.mean())
            result['top1_dist_meter_median'] = float(distances.median())
            output[name] = result
            msg = ' | '.join(f"R@{k}={result[f'recall@{k}'] * 100:.3f}%" for k in self.k_values)
            text = f'[{label}][{name}] {msg} | N={n}'
            print(text)
            if logger is not None:
                logger.info(text)
        # Preserve the former return shape when no validation partition exists.
        return output if 'val' in output else output.get('test', {})
