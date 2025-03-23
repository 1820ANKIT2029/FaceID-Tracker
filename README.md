# FaceID-Tracker

```bash
project-root/
│
├── frontend/                       # React frontend
│   ├── public/
│   │   └── models/                 # Pre-trained models (e.g., face-api.js)
│   ├── src/
│   │   ├── pages/
│   │   │    ├── FaceCapturePage.jsx
│   │   │    ├── Header.jsx
│   │   │    └── Table.jsx
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   └── vite.config.js
│
├── backend/                          # FastAPI
│   ├── ai-models/                    # Trained AI/ML models & code
│   │     ├── src/                    # Jupyter notebooks for experimentation
│   │     │   ├── data/               # Contains images - positive, negative, anchor, validation
│   │     │   ├── inference/
│   │     │   ├── notebooks/          # colab notebook - TEAM11.ipynb
│   │     │   ├── saved_models/       # model.keras
│   │     │   └── training/
│   │     ├── README.md
│   │     └── requirements.txt
│   ├── src/
│   │     ├─ db/
│   │     └── __init__.py
│   ├── __init__.py
│   ├── main.py
│   ├── README.md
│   └── requirement.txt
│
├── .gitignore
└── README.md
```