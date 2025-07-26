# DEPTH-MAP-CHROMO

This project provides an experimental UI for visualizing depth maps using several popular models.

## Backend

A Python backend exposes real depth estimation models.

### Installation

```bash
pip install -r backend/requirements.txt
```

### Running

```bash
python backend/server.py
```

The frontend expects the backend to be running at `http://localhost:8000`.

## Depth Pro

The official *Depth Pro* implementation is proprietary and not publicly available. This repository does not include that model. Please use one of the open models such as Depth Anything, MiDaS or Marigold instead.
