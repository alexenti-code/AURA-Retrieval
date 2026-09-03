# Memory Store

This directory holds the local plastic memory weight files for the MMI runtime.

In the current experimental setup:

- The stable cognitive core K is a cloud model accessed via a provider (e.g. router.ai).
- The plastic autobiographical memory Φ(t) is stored locally as a parameter file in this directory.
- MMI loads, exposes and assembles the memory context for each request.

## Files

Memory weight files are git-ignored. Each file represents a snapshot of the
plastic memory state for a specific instance.

Expected file types: .safetensors, .bin, .pt, .gguf

## Naming convention

    <instance_id>_<timestamp>.<ext>

Example: aura_instance_001_20260827T020000.safetensors
