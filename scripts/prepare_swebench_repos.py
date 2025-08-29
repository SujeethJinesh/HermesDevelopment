#!/usr/bin/env python3
"""Prepare SWE-bench repositories for hermetic execution.

This script prepares bare git mirrors (online phase) and creates
shallow worktrees (offline phase) for a given instances file.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_instances(instances_file: Path) -> List[str]:
    """Load instance IDs from file.
    
    Args:
        instances_file: Path to file with one instance ID per line
        
    Returns:
        List of instance IDs
    """
    if not instances_file.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_file}")
    
    instances = []
    with open(instances_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                instances.append(line)
    
    logger.info(f"Loaded {len(instances)} instances from {instances_file}")
    return instances


def parse_instance_id(instance_id: str) -> Dict[str, str]:
    """Parse instance ID to extract repo and commit.
    
    Args:
        instance_id: Instance ID like "django__django-11001"
        
    Returns:
        Dict with repo_owner, repo_name, issue_number
    """
    # Format: owner__repo-issue
    parts = instance_id.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid instance ID format: {instance_id}")
    
    repo_part = parts[0]
    issue_number = parts[1]
    
    repo_parts = repo_part.split("__")
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid repo format in instance ID: {instance_id}")
    
    return {
        "repo_owner": repo_parts[0],
        "repo_name": repo_parts[1],
        "issue_number": issue_number,
        "repo": f"{repo_parts[0]}/{repo_parts[1]}",
    }


def prepare_bare_mirrors(instances: List[str], mirrors_dir: Path, online: bool = True) -> Set[str]:
    """Prepare bare git mirrors for unique repositories.
    
    Args:
        instances: List of instance IDs
        mirrors_dir: Directory to store bare mirrors
        online: Whether to fetch from GitHub (online phase)
        
    Returns:
        Set of prepared repository names
    """
    mirrors_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract unique repositories
    repos = set()
    for instance_id in instances:
        info = parse_instance_id(instance_id)
        repos.add(info["repo"])
    
    logger.info(f"Found {len(repos)} unique repositories to prepare")
    
    for repo in sorted(repos):
        mirror_path = mirrors_dir / f"{repo.replace('/', '__')}.git"
        
        if mirror_path.exists():
            if online:
                # Update existing mirror
                logger.info(f"Updating mirror: {repo}")
                subprocess.run(
                    ["git", "fetch", "--all"],
                    cwd=mirror_path,
                    check=True,
                    capture_output=True
                )
            else:
                logger.info(f"Mirror exists (offline): {repo}")
        elif online:
            # Clone new mirror
            logger.info(f"Cloning mirror: {repo}")
            subprocess.run(
                [
                    "git", "clone", "--mirror",
                    f"https://github.com/{repo}.git",
                    str(mirror_path)
                ],
                check=True,
                capture_output=True
            )
        else:
            raise RuntimeError(f"Mirror not found (offline): {repo}")
    
    return repos


def create_worktrees(
    instances: List[str],
    mirrors_dir: Path,
    worktrees_dir: Path,
    dataset_dir: Path
) -> None:
    """Create shallow worktrees for each instance.
    
    Args:
        instances: List of instance IDs
        mirrors_dir: Directory with bare mirrors
        worktrees_dir: Directory for worktrees
        dataset_dir: SWE-bench dataset directory with metadata
    """
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    
    # Load instance metadata if available
    metadata_file = dataset_dir / "test.jsonl"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            for line in f:
                data = json.loads(line)
                metadata[data["instance_id"]] = data
    
    for instance_id in instances:
        info = parse_instance_id(instance_id)
        mirror_path = mirrors_dir / f"{info['repo'].replace('/', '__')}.git"
        worktree_path = worktrees_dir / instance_id
        
        if not mirror_path.exists():
            raise RuntimeError(f"Mirror not found: {mirror_path}")
        
        if worktree_path.exists():
            logger.info(f"Worktree exists: {instance_id}")
            continue
        
        # Get base commit from metadata if available
        base_commit = None
        if instance_id in metadata:
            base_commit = metadata[instance_id].get("base_commit")
        
        if not base_commit:
            # Try to find a reasonable commit (e.g., from issue creation date)
            # For now, use main/master HEAD as fallback
            logger.warning(f"No base commit for {instance_id}, using HEAD")
            base_commit = "HEAD"
        
        logger.info(f"Creating worktree: {instance_id} at {base_commit}")
        
        # Create worktree
        subprocess.run(
            [
                "git", "worktree", "add",
                "--detach",
                str(worktree_path),
                base_commit
            ],
            cwd=mirror_path,
            check=True,
            capture_output=True
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare SWE-bench repositories")
    parser.add_argument(
        "--instances-file",
        type=Path,
        required=True,
        help="File with instance IDs (one per line)"
    )
    parser.add_argument(
        "--mirrors-dir",
        type=Path,
        default=Path("data/repos/mirrors"),
        help="Directory for bare git mirrors"
    )
    parser.add_argument(
        "--worktrees-dir",
        type=Path,
        default=Path("scratch"),
        help="Directory for git worktrees"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/swebench_lite"),
        help="SWE-bench dataset directory"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Run in online mode (fetch from GitHub)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (use cached mirrors only)"
    )
    
    args = parser.parse_args()
    
    if args.online and args.offline:
        parser.error("Cannot specify both --online and --offline")
    
    if not args.online and not args.offline:
        # Default to online if not specified
        args.online = True
    
    # Load instances
    instances = load_instances(args.instances_file)
    
    # Phase 1: Prepare bare mirrors (online)
    if args.online:
        logger.info("Phase 1: Preparing bare mirrors (online)")
        repos = prepare_bare_mirrors(instances, args.mirrors_dir, online=True)
        logger.info(f"Prepared {len(repos)} repository mirrors")
    
    # Phase 2: Create worktrees (can be offline)
    logger.info("Phase 2: Creating worktrees")
    create_worktrees(
        instances,
        args.mirrors_dir,
        args.worktrees_dir,
        args.dataset_dir
    )
    logger.info(f"Created worktrees for {len(instances)} instances")
    
    # Write manifest
    manifest = {
        "instances": instances,
        "mirrors_dir": str(args.mirrors_dir),
        "worktrees_dir": str(args.worktrees_dir),
        "online": args.online,
    }
    manifest_file = args.worktrees_dir / "repos_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Wrote manifest to {manifest_file}")


if __name__ == "__main__":
    main()