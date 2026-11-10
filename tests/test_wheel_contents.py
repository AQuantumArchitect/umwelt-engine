"""Fail if the built wheel misses any file under src/ or a console script (#51)."""
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"umweltd", "umweltd-worker", "umweltctl", "umwelt-forge"}


def test_wheel_contains_all_src_files_and_scripts(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(tmp_path), str(ROOT)],
        check=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as z:
        names = set(z.namelist())
        ep = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
        entry_points = z.read(ep).decode()
    src = ROOT / "src"
    expected = {
        p.relative_to(src).as_posix()
        for p in src.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and ".egg-info" not in str(p) and p.suffix != ".pyc"
    }
    missing = sorted(expected - names)
    assert not missing, f"missing from wheel: {missing}"
    for s in SCRIPTS:
        assert f"{s} =" in entry_points, f"console script {s} not declared"
