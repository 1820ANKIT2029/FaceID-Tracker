# backend

Make and Activate Virtual env
```bash
python -m venv .venv
```
Linux, MacOS
```bash
source ./.venv/bin/activate 
```
Windows
```bash
.\.venv\Scripts\activate
```
Git Bash on Windows
```bash
source .venv/Scripts/activate
```

Download dependencies
```bash
pip install -r requirement.txt
```

Download model
```bash
pip install gdown
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O ./ai_models/saved_models/79999_iter.pth
gdown --id 18x_9yPi2pg4IePIaZlx1HFw7BP3nRoKy -O ./ai_models/saved_models/model.pth
```

Important Folder init
```bash
python -m ./ai_model/src
```

Upload image in Validation folder in ai_model/data/validation

Run the server
```bash
uvicorn main:app --reload
```