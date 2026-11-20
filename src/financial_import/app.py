from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .readers import ImportProblem, MAX_UPLOAD, preview_sheet
from .statements import FIELDS, ReviewedStatement, prepare_statement


WEB = Path(__file__).with_name("web")
app = FastAPI(title="Financial statement import")
app.mount("/financial-import-assets", StaticFiles(directory=WEB), name="financial-import-assets")


@app.middleware("http")
async def avoid_caching_private_responses(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/financial-statements/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def upload_page():
    return FileResponse(WEB / "index.html")


@app.get("/api/financial-statements/fields")
def supported_fields():
    return {name: asdict(field) for name, field in FIELDS.items()}


@app.get("/api/financial-statements/template.csv")
def download_template():
    return PlainTextResponse(
        "\ufeffitem,amount\r\n" + "".join(f"{name},\r\n" for name in FIELDS),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="financial-statements-template.csv"'},
    )


async def read_limited(request: Request, maximum: int) -> bytes:
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > maximum:
            raise HTTPException(413, "حجم الملف أو البيانات أكبر من الحد المسموح.")
        payload.extend(chunk)
    return bytes(payload)


@app.post("/api/financial-statements/preview")
async def preview(request: Request, format: Literal["csv", "xlsx"], separator: Literal["comma", "semicolon", "tab"] | None = None):
    payload = await read_limited(request, MAX_UPLOAD)
    delimiter = {"comma": ",", "semicolon": ";", "tab": "\t"}.get(separator)
    try:
        return await run_in_threadpool(preview_sheet, payload, format, delimiter)
    except ImportProblem as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, "فيه بيانات غير صالحة للقراءة في الملف. احفظ نسخة جديدة وجرّبها.") from exc


@app.post("/api/financial-statements/confirm")
async def confirm(request: Request):
    payload = await read_limited(request, 16 * 1024)
    try:
        review = ReviewedStatement.model_validate_json(payload)
    except ValidationError as exc:
        first = exc.errors(include_input=False, include_context=False, include_url=False)[0]
        message = first["msg"].removeprefix("Value error, ")
        if message.isascii():
            message = "راجع الفترة والعملة والمبالغ ووحدة كل بيان."
        raise HTTPException(422, message) from exc
    return prepare_statement(review)
