
# Correlation Scanner Bot

## Install & Run
```bash
cd /mnt/data/corr_bot
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt

# Edit symbols.txt to include your 100+ symbols (one per line)
python correlation_bot.py
```

### Outputs
- JSONL snapshots: `data/correlation_snapshots.jsonl`
- Latest snapshot:  `data/latest_correlation.json`
- Logs:             `data/correlation_bot.log`
