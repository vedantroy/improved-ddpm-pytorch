import typer

from typing import List
from pathlib import Path
import subprocess
import shutil
import os


def main(
    checkpoint_file: Path, output_dir: Path, extra_files: List[Path] = typer.Option([])
):
    assert checkpoint_file.is_file(), f"{checkpoint_file} is not a file"
    assert output_dir.is_dir(), f"{output_dir} is not a directory"

    for extra_file in extra_files:
        assert extra_file.is_file(), f"{extra_file} is not a file"

    folder_name = checkpoint_file.stem
    out_dir = output_dir / folder_name
    out_dir.mkdir()
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir()

    # Split checkpoint file into chunks so it can be uploaded to GH releases
    # (GH releases supports max 1GB file size)
    subprocess.run(
        ["split", "--bytes", "500M", str(checkpoint_file), str(chunks_dir) + "/"]
    )

    files = []
    for file in chunks_dir.glob("*"):
        files.append(file)
        shutil.move(file, chunks_dir / "..")

    shutil.rmtree(chunks_dir)
    files = [str(out_dir / str(file.name)) for file in files]

    for extra_file in extra_files:
        name = extra_file.name
        shutil.copy(extra_file, out_dir / name)

    concat_script = out_dir / "concat.sh"
    with open(concat_script, "w+") as f:
        f.write("#!/bin/sh\n")
        f.write(f"cat {' '.join(files)} > {out_dir}/{folder_name}.checkpoint\n")

    subprocess.check_call(["chmod", "+x", concat_script])


if __name__ == "__main__":
    typer.run(main)
