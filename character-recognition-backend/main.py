from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.post("/")
async def process(data: str = Form(...)):
    modelName = "char.model"
    import os
    from utils.helper import predict, checkAccuracy

    if not os.path.exists(modelName):
        from utils.helper import trainModel

        trainModel(modelName)
    # print(checkAccuracy(modelName))
    prediction = predict(modelName, data)
    return {"prediction": f"{prediction}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)


@app.get("/")
async def getreq():
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=5501)
