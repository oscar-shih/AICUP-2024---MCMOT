## Environment Settings
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## Reproduce Results
```bash
bash inference.sh --source_dir <path/to/dataset>
```
