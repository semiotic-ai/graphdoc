from datetime import datetime
import os
import random
import threading
from dotenv import load_dotenv
import litellm
import uuid
import logging
import dspy
import concurrent.futures
from graphdoc.config import dspy_lm_from_yaml
import queue

logger = logging.getLogger(__name__)

# global variables
api_call_count = 0
model_name = "gpt-4o-2024-08-06"
completion_tokens = 0
prompt_tokens = 0
total_tokens = 0
callback_lock = threading.Lock()

callback_queue = queue.Queue()
all_tasks_done = threading.Event()

def global_token_callback(kwargs, response, start_time, end_time, **callback_kwargs):
    data = {
        "model": response.get("model", "unknown"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
        "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
        "total_tokens": response.get("usage", {}).get("total_tokens", 0),
    }
    callback_queue.put(data)
    logger.info(f"Callback triggered, queued data, thread: {threading.current_thread().name}")

def math_chain(task_id, active_tasks):
    math = dspy.Predict("question -> answer: float")
    num_requests = random.randint(1, 3)
    logger.info(f"math_chain: num_requests: {num_requests}")
    for i in range(num_requests):
        value = random.randint(1, 10)
        result = math(question=f"What is 2+{value}? ID: {uuid.uuid4()}")
        logger.info(f"Task {task_id}, Request {i+1}: Result: {result.answer}")
    with callback_lock:
        active_tasks[0] -= 1
        if active_tasks[0] == 0:
            all_tasks_done.set()
    logger.info(f"Task {task_id} completed, remaining active tasks: {active_tasks[0]}")

def math_chain_multi():
    global api_call_count, model_name, completion_tokens, prompt_tokens, total_tokens
    
    with callback_lock:
        start_count = api_call_count
    logger.info(f"math_chain_multi started, initial api_call_count: {start_count}")

    num_tasks = random.randint(3, 7)
    logger.info(f"math_chain_multi: num_tasks: {num_tasks}")
    active_tasks = [num_tasks]
    all_tasks_done.clear()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(math_chain, i, active_tasks) for i in range(num_tasks)]
        logger.info(f"total futures: {len(futures)}")
        for future in concurrent.futures.as_completed(futures):
            future.result()
    
    all_tasks_done.wait()
    logger.info("All tasks completed, now draining callback queue...")
    
    callbacks_during_run = 0
    while True:
        try:
            data = callback_queue.get(timeout=2)
            with callback_lock:
                api_call_count += 1
                model_name = data["model"]
                completion_tokens += data["completion_tokens"]
                prompt_tokens += data["prompt_tokens"]
                total_tokens += data["total_tokens"]
            callbacks_during_run += 1
            callback_queue.task_done()
        except queue.Empty:
            logger.info("Queue empty after timeout, assuming all callbacks processed")
            break
    
    logger.info(f"math_chain_multi: model_name: {model_name}")
    logger.info(f"math_chain_multi: completion_tokens: {completion_tokens}")
    logger.info(f"math_chain_multi: prompt_tokens: {prompt_tokens}")
    logger.info(f"math_chain_multi: total_tokens: {total_tokens}")
    logger.info(f"math_chain_multi: total api_call_count: {api_call_count}")
    logger.info(f"math_chain_multi: callbacks during this run: {callbacks_during_run}")

def main():
    print("hello, world!")
    load_dotenv("../.env")
    if global_token_callback not in litellm.callbacks:
        litellm.callbacks.append(global_token_callback)
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    print(f"logging to file: {log_file}")
    logger.info("logging initialized")
    os.environ["LITELLM_LOG"] = "DEBUG"

    config_path = "/Users/denver/Documents/code/graph/graphdoc-mono/graphdoc/graphdoc/assets/configs/single_prompt_doc_generator_module_eval.yaml"
    dspy_lm_from_yaml(config_path)

    math_chain_multi()

if __name__ == "__main__":
    main()