backend

Download dependencies
```
pip install -r requirement.txt
```bash

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

Run the server
```bash
uvicorn main:app --reload
```