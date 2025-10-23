# Docling Parser with GPU Support

Convert PDF documents to Markdown and JSON using Docling with NVIDIA GPU acceleration.

## Quick Start

### Option 1: Remote PDF (URL)

```bash
docker-compose up
```

The default configuration will download and process: `https://arxiv.org/pdf/2408.09869`

### Option 2: Local PDF File

1. Create `input` folder and add your PDF:
```bash
mkdir -p input
cp your-document.pdf input/
```

2. Run the container:
```bash
docker-compose up
```

The script automatically detects and processes the first PDF in the `input/` folder.

## Output

Results are saved to `./output/`:
- `output.md` - Markdown format
- `output.json` - JSON format with full document structure

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA support

## Build from Scratch

```bash
docker-compose build --no-cache
docker-compose up
```

## Troubleshooting

### GPU Not Detected
Verify NVIDIA runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Use CPU Instead
Edit `docker-compose.yml` and remove GPU-related lines, then rebuild.
