from fastapi import FastAPI
from routers import vehicle, image  # <- sửa lại import

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

app.include_router(vehicle.router)
app.include_router(image.router)

