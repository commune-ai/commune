from fastapi import FastAPI, Query, Request

app = FastAPI()

@app.get("/test_with_arbitrary_query_params")
async def root(request: Request):
    return request.query_params._dict

@app.post("/test_with_dict")
async def root2(input: dict[str, str] = None):
    return input
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)