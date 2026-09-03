from pathlib import Path
import shutil
import subprocess

def git_clone(repository: str, branch: str, clone_path: str|Path = None, overwrite: bool = False) -> Path:
    """Clone a repository branch and run doxygen in the cloned directory."""
    
    if clone_path is None:
        clone_path = Path(repository).stem

    save_to_delete_later = ""
    if Path(clone_path).exists():
        if overwrite:
            save_to_delete_later = clone_path + "_BKUP"
            shutil.move(clone_path, save_to_delete_later)
        if not overwrite:
            print(f"REPOSITORY ALREADY EXISTS at {clone_path}")
            return
            
    print(f"Cloning {repository} on branch {branch}...")
    clone_result = subprocess.run(
        ["git", "clone", "-b", branch, repository, clone_path],
        capture_output=True,
        text=True,
    )
    
    if clone_result.returncode != 0:
        print(f"git clone error: {clone_result.stderr.strip()}")
        if save_to_delete_later:
            shutil.move(save_to_delete_later, clone_path)
        raise RuntimeError(f"Failed to clone '{repository}' on branch {branch}.")

    if save_to_delete_later:
        shutil.rmtree(save_to_delete_later)

    return Path(clone_path)


def run_doxygen(docpath: str|Path):

    print(f"Running doxygen in {docpath}...")
    doxygen_result = subprocess.run(
        ["doxygen"],
        cwd=docpath,
        capture_output=True,
        text=True,
    )
    if doxygen_result.returncode != 0:
        print(f"doxygen error: {doxygen_result.stderr.strip()}")
        raise RuntimeError(f"doxygen command failed")

    return Path(docpath)