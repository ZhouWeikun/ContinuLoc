import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainer_depends.config.parser import _expand_selected_scene_config, get_parse


class ManifestSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'drone').mkdir()
        for name in ('sat.json', 'uav.json', 'geo.csv', 'sat.tif'):
            (self.root / name).touch()
        entry = dict(status='prepared', p_satinfo_json='sat.json',
                     p_uavinfo_json='uav.json', p_uav_geocsv='geo.csv',
                     satellite_filepaths=['sat.tif'], uavimgs_dir='drone')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(dataset_name='visloc',
            default_scene_names=['visloc_03', 'visloc_04'],
            scene_registry={'visloc_03': entry, 'visloc_04': entry,
                            'visloc_07': dict(status='unprepared')})))
        self.settings = dict(dataset_manifest=str(self.manifest), dataset_name='visloc')

    def test_aliases_paths_order_and_repeat_expansion(self):
        settings = dict(self.settings, selected_scene_names=[4, 'visloc_03'])
        original = copy.deepcopy(settings)
        result = _expand_selected_scene_config(settings)
        self.assertEqual(settings, original)
        self.assertEqual([s['name'] for s in result['scenes']], ['visloc_04', 'visloc_03'])
        self.assertEqual(result['scenes'][0]['p_uav_geocsv'], str(self.root / 'geo.csv'))
        self.assertEqual(result['scenes'][0]['satellite_filepaths'], [str(self.root / 'sat.tif')])
        self.assertEqual(_expand_selected_scene_config(result), result)

    def test_invalid_selections(self):
        for names in ([], ['99'], ['07'], ['03', 3]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                _expand_selected_scene_config(dict(self.settings, selected_scene_names=names))
        (self.root / 'sat.tif').unlink()
        with self.assertRaises(FileNotFoundError):
            _expand_selected_scene_config(self.settings)

    def test_legacy_configs(self):
        old = {'scenes': [{'name': 'zurich'}]}
        self.assertEqual(_expand_selected_scene_config(old), old)
        registry = dict(scene_registry={'zurich': {'name': 'zurich'}}, selected_scene_name='zurich')
        self.assertEqual(_expand_selected_scene_config(registry)['scenes'], old['scenes'])

    def test_cli_overrides_yaml_selection(self):
        config = self.root / 'config.yaml'
        for yaml_selection, cli, expected in [
            ({'selected_scene_names': ['03', '04']}, ['--selected_scene_name', '04'], ['visloc_04']),
            ({'selected_scene_name': '03'}, ['--selected_scene_names', '04', '03'], ['visloc_04', 'visloc_03']),
        ]:
            import yaml
            config.write_text(yaml.safe_dump({'scenes_setting': dict(self.settings, **yaml_selection)}))
            with patch('sys.argv', ['train', '--p_yaml', str(config)] + cli):
                opt = get_parse(print_summary=False)
            self.assertEqual([s['name'] for s in opt.scenes_setting['scenes']], expected)


if __name__ == '__main__':
    unittest.main()
