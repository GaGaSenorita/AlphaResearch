# Local macOS runtime

The recovered AlphaResearch repository runs locally without an NVIDIA GPU.
LLM generation is remote, while Qlib evaluation and the GPyTorch surrogate run
on the Mac CPU.

## Activate and verify

```bash
cd "/Users/senorita/Desktop/Alpha Mining Report/AlphaResearch"
conda activate AlphaResearch
bash scripts/check_local_setup.sh
```

The Conda activation hook reads the LLM credential from the macOS Keychain. It
does not store the secret in this repository or in a YAML file.

## Run continual discovery

Use a short run first:

```bash
bash scripts/run_local_continuous_discovery.sh 2 999
```

Run 100 rounds for a real seed:

```bash
bash scripts/run_local_continuous_discovery.sh 100 42
```

The first argument is the cumulative target round and the second is the seed.
The launcher starts the local FFO backend when needed and uses `caffeinate` to
keep macOS awake. Completed rounds are resumable from the canonical run folder.

## Services

```bash
ppo status
ppo logs backend -n 100
ppo stop backend
```

The Qlib provider is stored at
`../quantaalpha_qlib_csi300/cn_data`, outside the Git repository but inside the
`Alpha Mining Report` folder. The standard Qlib path
`~/.qlib/qlib_data/cn_data` points to the same provider.

