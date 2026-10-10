"""Field (plot) management and device registration."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Response, status

from ...db import repository as repo
from ..dependencies import get_conn
from ..schemas import DeviceIn, DeviceOut, FieldCreate, FieldOut, FieldUpdate

router = APIRouter(prefix="/api/v1", tags=["fields"])


@router.post("/fields", response_model=FieldOut, status_code=status.HTTP_201_CREATED, summary="Create a field")
def create_field(payload: FieldCreate, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    return repo.create_field(conn, payload.to_db())


@router.get("/fields", response_model=list[FieldOut], summary="List fields")
def list_fields(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    return repo.list_fields(conn)


@router.get("/fields/{field_id}", response_model=FieldOut, summary="Get one field")
def get_field(field_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    return repo.get_field(conn, field_id)


@router.patch("/fields/{field_id}", response_model=FieldOut, summary="Update a field (partial)")
def update_field(field_id: int, payload: FieldUpdate, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    return repo.update_field(conn, field_id, payload.to_db())


@router.delete("/fields/{field_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a field and its data")
def delete_field(field_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> Response:
    repo.delete_field(conn, field_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/devices", response_model=DeviceOut, status_code=status.HTTP_201_CREATED, tags=["iot"],
             summary="Register (or re-assign) an IoT sensor node to a field")
def register_device(payload: DeviceIn, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    return repo.register_device(conn, payload.device_id, payload.field_id, payload.description)


@router.get("/devices", response_model=list[DeviceOut], tags=["iot"], summary="List registered devices")
def list_devices(field_id: int | None = None, conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    return repo.list_devices(conn, field_id)
