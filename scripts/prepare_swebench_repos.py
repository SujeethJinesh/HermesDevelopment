#!/usr/bin/env python3
"""Prepare local git mirrors for hermetic SWE-bench Lite evaluation.

This script runs ONLINE once to create local mirrors/bundles of all repositories
needed for a SWE-bench Lite slice. During hermetic evaluation, these mirrors
are used offline to create fresh worktrees at specific commits.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from datasets import load_dataset


def get_unique_repos(instances_file: Path) -> Set[Tuple[str, str]]:
    """Extract unique (repo, base_commit) pairs from instance slice."""
    # Load instance IDs from file
    instance_ids = []
    with open(instances_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                instance_ids.append(line)
    
    # Load SWE-bench Lite dataset
    print(f"Loading SWE-bench/SWE-bench_Lite to get repo info...")
    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    
    # Find instances and collect unique repos
    unique_repos = set()
    for instance in dataset:
        if instance["instance_id"] in instance_ids:
            repo = instance["repo"]
            base_commit = instance["base_commit"]
            unique_repos.add((repo, base_commit))
    
    return unique_repos


def create_mirror(repo: str, base_commit: str, mirrors_dir: Path, bundles_dir: Path) -> Dict:
    """Create local mirror and optionally bundle for a repo."""
    org, name = repo.split("/")
    mirror_path = mirrors_dir / f"{org}__{name}.git"
    bundle_path = bundles_dir / f"{org}__{name}.bundle"
    
    result = {
        "repo": repo,
        "mirror_path": str(mirror_path),
        "bundle_path": str(bundle_path),
        "base_commit": base_commit,
        "status": "pending"
    }
    
    try:
        # Clone as mirror if doesn't exist
        if not mirror_path.exists():
            print(f"  Cloning mirror for {repo}...")
            subprocess.run(
                ["git", "clone", "--mirror", f"https://github.com/{repo}.git", str(mirror_path)],
                check=True, capture_output=True, text=True
            )
        else:
            print(f"  Mirror exists for {repo}, fetching updates...")
            subprocess.run(
                ["git", "--git-dir", str(mirror_path), "fetch", "--all", "--tags"],
                check=True, capture_output=True, text=True
            )
        
        # Verify base_commit exists
        try:
            subprocess.run(
                ["git", "--git-dir", str(mirror_path), "rev-parse", base_commit],
                check=True, capture_output=True, text=True
            )
            print(f"    ✓ Commit {base_commit[:8]} exists")
        except subprocess.CalledProcessError:
            # Try to fetch the specific commit
            print(f"    ! Commit {base_commit[:8]} not found, fetching...")
            subprocess.run(
                ["git", "--git-dir", str(mirror_path), "fetch", "origin", base_commit],
                capture_output=True, text=True
            )
            # Verify again
            subprocess.run(
                ["git", "--git-dir", str(mirror_path), "rev-parse", base_commit],
                check=True, capture_output=True, text=True
            )
            print(f"    ✓ Commit {base_commit[:8]} fetched")
        
        # Create bundle for faster offline restore
        if not bundle_path.exists():
            print(f"    Creating bundle...")
            subprocess.run(
                ["git", "--git-dir", str(mirror_path), "bundle", "create", str(bundle_path), "--all"],
                check=True, capture_output=True, text=True
            )
            print(f"    ✓ Bundle created at {bundle_path}")
        
        result["status"] = "success"
        
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare repo mirrors for hermetic SWE-bench evaluation")
    parser.add_argument("--instances_file", type=str, default="configs/swebench_lite_slice20.txt",
                        help="File with instance IDs to prepare")
    parser.add_argument("--out", type=str, default="data/repos_manifest.json",
                        help="Output manifest file")
    parser.add_argument("--mirrors_dir", type=str, default="data/repos/mirrors",
                        help="Directory for git mirrors")
    parser.add_argument("--bundles_dir", type=str, default="data/repos/bundles",
                        help="Directory for git bundles")
    args = parser.parse_args()
    
    instances_file = Path(args.instances_file)
    mirrors_dir = Path(args.mirrors_dir)
    bundles_dir = Path(args.bundles_dir)
    manifest_path = Path(args.out)
    
    if not instances_file.exists():
        print(f"Error: {instances_file} not found")
        sys.exit(1)
    
    # Create directories
    mirrors_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing repos for instances in {instances_file}")
    print(f"Mirrors will be stored in {mirrors_dir}")
    print(f"Bundles will be stored in {bundles_dir}")
    
    # Get unique repos needed
    unique_repos = get_unique_repos(instances_file)
    print(f"\nFound {len(unique_repos)} unique repo/commit pairs to prepare")
    
    # Create mirrors and bundles
    manifest = {
        "instances_file": str(instances_file),
        "repos": []
    }
    
    for repo, base_commit in sorted(unique_repos):
        print(f"\nProcessing {repo} @ {base_commit[:8]}...")
        result = create_mirror(repo, base_commit, mirrors_dir, bundles_dir)
        manifest["repos"].append(result)
    
    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    successful = sum(1 for r in manifest["repos"] if r["status"] == "success")
    failed = sum(1 for r in manifest["repos"] if r["status"] == "failed")
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"Manifest saved to {manifest_path}")
    
    if failed > 0:
        print("\nFailed repos:")
        for r in manifest["repos"]:
            if r["status"] == "failed":
                print(f"  - {r['repo']}: {r.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("\n✅ All repos prepared successfully for hermetic evaluation")


if __name__ == "__main__":
    main()