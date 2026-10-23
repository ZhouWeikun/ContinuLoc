"""Synthetic checks for location aggregation vs joint pose ranking."""
import torch
from evaluate_xy import rank_rotation_candidates

angles = torch.deg2rad(torch.tensor([0., 90., 180.]))
# xy0 has one sharp peak; xy1 has support at all angles.
responses = torch.tensor([[[0.99, 0.8, 0.1], [-1., 0.82, 0.2], [-1., 0.81, 0.3]]])
idx, heading, scores = rank_rotation_candidates(responses, angles, 3, 'rot_weighting_argmax', 1.)
assert idx[0, 0].item() == 1
assert torch.allclose(torch.rad2deg(heading[0, 0]), torch.tensor(-90.))
expected = torch.log(torch.exp(-torch.sqrt(2 - 2 * responses.double())).sum(1))
assert torch.allclose(scores, expected.gather(1, idx), atol=1e-6)
assert len(set(idx[0].tolist())) == 3
idx, heading, scores = rank_rotation_candidates(responses, angles, 3, 'rot_xy_argmax', .02)
assert idx.tolist() == [[0, 1, 1]]
assert torch.allclose(scores, torch.tensor([[.99, .82, .81]]))
assert torch.allclose(torch.rad2deg(heading[0, :2]), torch.tensor([0., -90.]))
idx, _, _ = rank_rotation_candidates(responses, angles, 3, 'rot_argmax', .02)
assert idx.tolist() == [[0, 1, 2]]
idx, heading, scores = rank_rotation_candidates(responses, angles, 3, 'rot_sum_argmax', 0.)
assert idx.tolist() == [[1, 2, 0]]
assert torch.allclose(scores, torch.tensor([[2.43, .6, -1.01]]), atol=1e-6)
assert torch.allclose(torch.rad2deg(heading[0, :1]), torch.tensor([-90.]))
idx2, heading2, scores2 = rank_rotation_candidates(responses, angles, 3, 'rot_sum_argmax', 99.)
assert torch.equal(idx, idx2) and torch.equal(heading, heading2) and torch.equal(scores, scores2)
print('Aggregation scores, discrete headings, joint duplicate xy and legacy ranking passed.')
