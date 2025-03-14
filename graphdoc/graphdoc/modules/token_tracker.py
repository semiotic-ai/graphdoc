# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import queue
import logging
import threading

# external packages

# internal packages

# logging
log = logging.getLogger(__name__)

class TokenTracker: 

    def __init__(self):
        self.model_name = ""
        self.api_call_count = 0 
        self.completion_tokens = 0
        self.prompt_tokens = 0 
        self.total_tokens = 0 
        self.active_tasks = 0
        self.callback_lock = threading.Lock()
        self.callback_queue = queue.Queue()
        self.all_tasks_done = threading.Event()

    def clear(self):
        self.api_call_count = 0 
        self.model_name = ""
        self.completion_tokens = 0
        self.prompt_tokens = 0 
        self.total_tokens = 0 
        self.active_tasks = 0

    def stats(self):
        return {
            "model_name": self.model_name,
            "api_call_count": self.api_call_count,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        }

    def global_token_callback(self, kwargs, response, start_time, end_time, **callback_kwargs):
        data = {
            "model": response.get("model", "unknown"),
            "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
            "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
            "total_tokens": response.get("usage", {}).get("total_tokens", 0),
        }
        self.callback_queue.put(data)
        log.info(f"Callback triggered, queued data, thread: {threading.current_thread().name}")