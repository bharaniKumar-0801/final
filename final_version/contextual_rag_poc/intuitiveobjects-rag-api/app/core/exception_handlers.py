from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.schema.response import APIResponse

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    print(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(success=False, message=exc.detail, data=None).dict()
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=APIResponse(success=False, message="Validation Error", data=exc.errors()).dict()
    )
