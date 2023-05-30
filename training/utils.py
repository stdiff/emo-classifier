from pathlib import Path
import tarfile

def _target_is_under_base_directory(base_directory: Path, target_path: Path):
    return any(base_directory.resolve() == parent.resolve() for parent in target_path.parents)

def tarball_extract_under_dir(tarball_path: Path, target_dir: Path):
    """
    Tarfile.extractall() with sanity check of file paths (cf. CVE-2007-4559)

    :param tarball_path: Path to the tarball
    :param target_dir: all files must be extracted under this directory
    """
    with tarfile.open(tarball_path, "r:gz") as tar_ball:
        for file in tar_ball.getmembers():
            file_path = target_dir / file.name
            if _target_is_under_base_directory(target_dir, file_path):
                raise Exception(f"Unexpected file path: {file_path.resolve()}")

        tar_ball.extractall(target_dir)

