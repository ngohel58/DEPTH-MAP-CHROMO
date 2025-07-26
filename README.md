# DEPTH-MAP-CHROMO

This project provides an experimental UI for visualizing depth maps using several popular models.
The latest `index.html` runs everything directly in the browser using **ONNX Runtime WebGL**. Download
the ONNX model files and place them in the `models/` directory or adjust the paths in the code.

The previous Python backend is still included for reference but is no longer required.

### Optional backend (legacy)

If you wish to run the old FastAPI backend, install the dependencies and start the server:

```bash
pip install -r backend/requirements.txt
python backend/server.py
```

## Depth Pro

The official *Depth Pro* implementation is proprietary and not publicly available. This repository does not include that model. Please use one of the open models such as Depth Anything, MiDaS or Marigold instead.
