"""."""
from pathlib import Path

import pytest

from trackeval.cli.run_kitti import run


def test_run_kitti(files_dir: Path, capsys: pytest.CaptureFixture) -> None:
    """."""
    run([
        '--GT_FOLDER', str(files_dir / 'data/gt/kitti/kitti_2d_box_train'),
        '--TRACKERS_FOLDER', str(files_dir / 'data/trackers/kitti/kitti_2d_box_train'),
        '--CLASSES_TO_EVAL', 'pedestrian',
        '--OUTPUT_FOLDER', '.',
        '--SPLIT_TO_EVAL', 'val'
    ])
    stdout = capsys.readouterr().out
    ref_hota = 'COMBINED                           49.137    45.145    54.027    56.285    56.51     58.816    70.106    74.881    55.041    74.747    66.321    49.573'
    ref_clear = 'COMBINED                           48.593    70.645    49.755    74.679    74.977    54.412    35.294    10.294    26.671    2442      828       815       38        37        24        7         80'
    ref_ident = 'COMBINED                           68.883    68.746    69.021    2248      1022      1009'
    assert ref_hota in stdout
    assert ref_clear in stdout
    assert ref_ident in stdout

