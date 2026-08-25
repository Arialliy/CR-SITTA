# Model weights

Download the official checkpoints linked in `configs/protocol.yaml` into this
directory. Checkpoint binaries are intentionally ignored by git; their SHA-256
digests are frozen in the protocol after download and verification.

Reproducible download pattern:

```bash
wget --tries=8 --timeout=30 --output-document=<destination> \
  'https://drive.usercontent.google.com/download?id=<file-id>&export=download&confirm=t'
```

| Dataset | File ID | Local filename | SHA-256 |
| --- | --- | --- | --- |
| IRSTD-1k | `1agnCjpJJa3J3-Aw8XqDKtpcTA6xDuHO4` | `IRSTD-1k_MSHNet_NSFPN.pkl` | `9b22e5dfa82e033cb85009d74db5db637cde1dc186fd302f5aaba4515dee2d6b` |
| NUAA-SIRST | `17zgfkbkPdLGyOLDz_MFNbUQI9J2bmgiI` | `NUAA-SIRST_MSHNet_NSFPN.pkl` | `ed76bc3b0b24b7986c69fd01e002aa911ad6c0528702f7e493029ea01a795a43` |
