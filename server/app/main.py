import base64
import json
import os
import shutil
import tempfile
from typing import Annotated, List, Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
from starlette.status import HTTP_401_UNAUTHORIZED

from .asr import LOCAL_SETUP, local_asr_available, scribe
from .copilot import CopilotService, create_copilot_router
from .models import (
    DownloadModelTask,
    LanguageDoesNotExist,
    ModelDoesNotExist,
    ModelNotDownloaded,
    ModelTypeNotSupported,
    models,
)
from .otio import Segment, convert_otio
from .tasks import TaskNotFoundError, tasks
from .transcribe import TranscriptionState, TranscriptionTask, process_audio

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_TOKEN = base64.b64encode(os.urandom(64)).decode()


def token_auth(request: Request):
    authorization: str = request.headers.get("Authorization")
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Not authorized")
    return authorization


app.state.copilot = CopilotService()
app.include_router(create_copilot_router(token_auth))


@app.on_event("startup")
def startup_event():
    print(json.dumps({"msg": "server_started", "token": AUTH_TOKEN}), flush=True)


@app.post("/tasks/start_transcription/")
async def start_transcription(
    background_tasks: BackgroundTasks,
    transcription_model: str | None = None,
    provider: Literal["vosk", "scribe"] = "vosk",
    language_code: Annotated[str | None, Query(pattern=r"^[a-z]{2,3}$")] = None,
    diarize_max_speakers: Annotated[int | None, Query(ge=1, le=32)] = None,
    diarize: bool = False,
    file: UploadFile = File(...),
    fileName: str = Form(...),
    auth: str = Depends(token_auth),
):
    if provider == "scribe" and not scribe.is_configured():
        raise HTTPException(status_code=503, detail="ElevenLabs is not configured on this server.")
    if provider == "vosk":
        if not transcription_model:
            raise HTTPException(status_code=422, detail="Choose a local transcription model.")
        if not local_asr_available():
            raise HTTPException(status_code=503, detail=LOCAL_SETUP)
        if not models.get_model_description(transcription_model).is_downloaded():
            raise ModelNotDownloaded()

    # The background job must outlive FastAPI's request-owned upload.
    audio = tempfile.TemporaryFile()
    try:
        await run_in_threadpool(shutil.copyfileobj, file.file, audio)
        audio.seek(0)
    except BaseException:
        audio.close()
        raise
    task = tasks.add(
        TranscriptionTask(
            file.filename or fileName,
            TranscriptionState.QUEUED,
        )
    )
    background_tasks.add_task(
        process_audio,
        task,
        audio,
        fileName,
        provider,
        transcription_model,
        language_code,
        diarize,
        diarize_max_speakers,
    )
    return task


@app.get("/transcription/capabilities/")
async def transcription_capabilities(auth: str = Depends(token_auth)):
    return {"scribe": scribe.is_configured(), "vosk": local_asr_available()}


@app.post("/tasks/download_model/")
async def download_model(
    background_tasks: BackgroundTasks,
    model_id: str,
    auth: str = Depends(token_auth),
):
    task = tasks.add(DownloadModelTask(model_id))
    background_tasks.add_task(models.download, model_id, task.uuid)
    return task


# FIXME: this needs to be removed / put behind proper auth for security reasons
@app.get("/tasks/list/")
async def list_tasks(auth: str = Depends(token_auth)):
    return sorted(tasks.list(), key=lambda x: x.uuid)


@app.get("/tasks/{task_uuid}/")
async def get_task(task_uuid: str, auth: str = Depends(token_auth)):
    return tasks.get(task_uuid)


@app.delete("/tasks/{task_uuid}/")
async def remove_task(task_uuid: str, auth: str = Depends(token_auth)):
    return tasks.delete(task_uuid)


@app.get("/models/available")
async def get_all_models(auth: str = Depends(token_auth)):
    return models.available


@app.post("/models/delete")
async def delete_model(model_id: str, auth: str = Depends(token_auth)):
    models.delete(model_id)
    return PlainTextResponse("", status_code=200)


@app.get("/models/downloaded")
async def get_downloaded_models(auth: str = Depends(token_auth)):
    return models.downloaded


@app.post("/util/otio/convert")
async def convert_otio_http(
    name: str,
    adapter: str,
    timeline: List[Segment],
    auth: str = Depends(token_auth),
):
    converted = convert_otio(timeline, name, adapter)
    return PlainTextResponse(converted)


@app.exception_handler(TaskNotFoundError)
async def task_not_found_error_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=404)


@app.exception_handler(LanguageDoesNotExist)
async def language_does_not_exist_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=404)


@app.exception_handler(ModelDoesNotExist)
async def model_does_not_exist_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=404)


@app.exception_handler(ModelNotDownloaded)
async def model_not_downloaded_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=412)


@app.exception_handler(ModelTypeNotSupported)
async def model_type_not_supported(request, exc):
    return PlainTextResponse(str(exc), status_code=412)
