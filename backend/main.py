from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Message" : "Server is running..."}