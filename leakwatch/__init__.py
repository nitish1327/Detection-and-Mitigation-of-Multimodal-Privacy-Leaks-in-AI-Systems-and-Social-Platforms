"""LeakWatch privacy-preserving middleware package."""

import os
import shutil
from pathlib import Path

from .version import __version__  # noqa: F401


def _ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return

    import sys
    scripts_dir = Path(sys.prefix) / "Scripts"
    if (scripts_dir / "ffmpeg.exe").exists() or (scripts_dir / "ffmpeg").exists():
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
        return

    root = Path(__file__).resolve().parent.parent
    shim_dir = root / ".ffmpeg_shim"
    if (shim_dir / "ffmpeg.exe").exists() or (shim_dir / "ffmpeg").exists():
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        return

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_exe = get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            ffmpeg_path = Path(ffmpeg_exe)
            parent = ffmpeg_path.parent
            target_exe = parent / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not target_exe.exists():
                try:
                    shutil.copyfile(ffmpeg_path, target_exe)
                    target_exe.chmod(0o755)
                except Exception:
                    pass
            if target_exe.exists():
                os.environ["PATH"] = str(parent) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_ensure_ffmpeg_on_path()
