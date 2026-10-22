"""Start an installed local Ollama server when the project launcher needs it."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlparse

import requests


def ensure_local_llm(config) -> bool:
    if config.llm.provider != 'ollama':
        return True
    base = (config.llm.api_base or 'http://127.0.0.1:11434').rstrip('/')
    if urlparse(base).hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('Cấu hình Ollama của dự án này phải dùng máy cục bộ.')

    def ready():
        try:
            return requests.get(base+'/api/tags',timeout=2).ok
        except requests.RequestException:
            return False

    if ready():
        return True
    local = Path(os.environ.get('LOCALAPPDATA','')) / 'Programs'
    candidates = [shutil.which('ollama'), local/'OllamaRAG'/'ollama.exe', local/'Ollama'/'ollama.exe']
    binary = next((str(p) for p in candidates if p and Path(p).is_file()),None)
    if not binary:
        print('[RAG] Chưa tìm thấy Ollama. Xem docs/RAG.md để cài dịch vụ cục bộ.')
        return False
    env = os.environ.copy()
    env['OLLAMA_HOST'] = base
    env['OLLAMA_NO_CLOUD'] = '1'
    config.knowledge_base.index_dir.mkdir(parents=True,exist_ok=True)
    with (config.knowledge_base.index_dir/'ollama-server.log').open('ab') as log:
        subprocess.Popen([binary,'serve'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        if ready():
            return True
        time.sleep(.5)
    print('[RAG] Ollama chưa sẵn sàng; xem outputs/rag_index/ollama-server.log.')
    return False
