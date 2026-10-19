#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import logging
import pickle  # nosec B403: Safe usage for internal serialization only
from multiprocessing.synchronize import Event as MpEvent
from queue import Queue
from typing import Any

import torch

from lerobot.utils.constants import OBS_IMAGE
from lerobot.utils.transition import Transition

from . import services_pb2

TransferState = services_pb2.TransferState

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def bytes_buffer_size(buffer: io.BytesIO) -> int:
    buffer.seek(0, io.SEEK_END)
    result = buffer.tell()
    buffer.seek(0)
    return result


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    bytes_buffer: io.BytesIO = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(bytes_buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(iterator, queue: Queue | None, shutdown_event: MpEvent, log_prefix: str = ""):
    bytes_buffer = io.BytesIO()
    step = 0

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if shutdown_event.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0
        elif item.transfer_state == TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")
        elif item.transfer_state == TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert model state dict to flat array for transmission"""
    bytes_buffer = io.BytesIO()

    torch.save(state_dict, bytes_buffer)

    return bytes_buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return torch.load(bytes_buffer, weights_only=True)


def python_object_to_bytes(python_object: Any) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    obj = pickle.load(bytes_buffer)  # nosec B301: Safe usage of pickle.load
    # Add validation checks here
    return obj


# Compact transition format: images in [0, 1] are sent as uint8 and a `next_state` entry equal to the next
# transition's `state` entry is sent once. About 8x smaller for image observations, which matters when the
# learner runs on another machine.
COMPACT_TRANSITIONS_FORMAT = "lerobot.compact_transitions.v1"
_SAME_AS_NEXT_STATE = "__same_as_next_transition_state__"
_UINT8_KEY = "__uint8__"


def _encode_value(key: str, value: Any) -> Any:
    if (
        key.startswith(OBS_IMAGE)
        and isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and value.numel() > 0
        and value.min() >= 0
        and value.max() <= 1
    ):
        # Quantization error is at most 1/510, below what the camera resolves.
        return {
            _UINT8_KEY: (value * 255).round().to(torch.uint8),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict) and _UINT8_KEY in value:
        return value[_UINT8_KEY].to(getattr(torch, value["dtype"])) / 255
    return value


def _pack_transitions(transitions: list[Transition]) -> dict[str, Any]:
    packed = []
    for i, transition in enumerate(transitions):
        following_state = transitions[i + 1]["state"] if i + 1 < len(transitions) else {}
        next_state = {}
        for key, value in transition["next_state"].items():
            following = following_state.get(key)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(following, torch.Tensor)
                and value.shape == following.shape
                and value.dtype == following.dtype
                and torch.equal(value, following)
            ):
                next_state[key] = _SAME_AS_NEXT_STATE
            else:
                next_state[key] = _encode_value(key, value)
        item = dict(transition)
        item["state"] = {key: _encode_value(key, value) for key, value in transition["state"].items()}
        item["next_state"] = next_state
        packed.append(item)
    return {"format": COMPACT_TRANSITIONS_FORMAT, "transitions": packed}


def _unpack_transitions(packed: list[dict[str, Any]]) -> list[Transition]:
    transitions: list[dict[str, Any]] = []
    for item in packed:
        unpacked = dict(item)
        unpacked["state"] = {key: _decode_value(value) for key, value in item["state"].items()}
        transitions.append(unpacked)
    for i, unpacked in enumerate(transitions):
        unpacked["next_state"] = {
            key: transitions[i + 1]["state"][key]
            if isinstance(value, str) and value == _SAME_AS_NEXT_STATE
            else _decode_value(value)
            for key, value in unpacked["next_state"].items()
        }
    return transitions  # type: ignore[return-value]


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    data = torch.load(bytes_buffer, weights_only=True)
    if isinstance(data, dict) and data.get("format") == COMPACT_TRANSITIONS_FORMAT:
        return _unpack_transitions(data["transitions"])
    return data  # plain list (older senders)


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    bytes_buffer = io.BytesIO()
    torch.save(_pack_transitions(transitions), bytes_buffer)
    return bytes_buffer.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # Applies to ALL methods in ALL services
                "retryPolicy": {
                    "maxAttempts": max_attempts,  # Max retries (total attempts = 5)
                    "initialBackoff": initial_backoff,  # First retry after 0.1s
                    "maxBackoff": max_backoff,  # Max wait time between retries
                    "backoffMultiplier": backoff_multiplier,  # Exponential backoff factor
                    "retryableStatusCodes": [
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ],  # Retries on network failures
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)

    retries_option = 1 if enable_retries else 0

    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", retries_option),
        ("grpc.service_config", service_config_json),
    ]
