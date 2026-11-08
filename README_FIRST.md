# Storage Location

## Where are the working files?

These MLX8 course project files have been relocated from the original working
location on unraid1 to gpu1 for long-term archival storage.

### Previous location (unraid1 -- home lab)

```
/mnt/cache/__tmp/workspace/_github/{owner}/{repo}/
```

- **Host:** unraid1 (192.168.50.243, Unraid 7.x)
- **Drive:** 4 TB NVMe cache (`/mnt/cache`)
- **Status:** Files removed after verified sync to gpu1

### Current location (gpu1 -- campus)

```
/mnt/spool/bkup/unraid1-workspace/_github/{owner}/{repo}/
```

- **Host:** gpu1 (192.168.50.244, Ubuntu 24.04)
- **Pool:** spool -- 4x22T HDD RAIDZ1 (ZFS)
- **Access:** `ssh gpu1` via Tailscale

### What is NOT in git

The `.charles/` directories contain large model checkpoints, training outputs,
and datasets (tens to hundreds of GB per repo). These are gitignored and exist
only on disk -- they were synced to gpu1 via rsync on 2026-11-08.

### To restore working copies

```bash
# From gpu1, rsync back to any machine:
rsync -aHz gpu1:/mnt/spool/bkup/unraid1-workspace/_github/charles-cai/{repo}/ ./{repo}/

# Then recreate the Python environment:
cd {repo}
uv sync   # or: pip install -r requirements.txt
```
