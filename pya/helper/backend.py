from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pya.backend.base import BackendBase
    from pya.backend.PyAudio import PyAudioBackend
    from pya.backend.Jupyter import JupyterBackend


def get_server_info():
    import re
    import json
    import requests
    import ipykernel
    import notebook.notebookapp
    kernel_id = re.search(
        "kernel-(.*).json",
        ipykernel.connect.get_connection_file()
    ).group(1)
    servers = notebook.notebookapp.list_running_servers()
    for s in servers:
        response = requests.get(
            requests.compat.urljoin(s["url"], "api/sessions"),
            params={"token": s.get("token", "")}
        )
        for n in json.loads(response.text):
            if n["kernel"]["id"] == kernel_id:
                return s
    return None


def try_pyaudio_backend(**kwargs) -> Optional["PyAudioBackend"]:
    try:
        from pya.backend.PyAudio import PyAudioBackend
        return PyAudioBackend(**kwargs)
    except ImportError:
        return None


def try_jupyter_backend(port, **kwargs) -> Optional["JupyterBackend"]:
    import os
    from pya.backend.Jupyter import JupyterBackend
    server_info = get_server_info()
    if server_info is None or (server_info['hostname'] in ['localhost', '127.0.0.1'] and not force_webaudio):
        return None  # use default local backend
    if os.environ.get('BINDER_SERVICE_HOST'):
        return JupyterBackend(port=port, proxy_suffix=f"/proxy/{port}", **kwargs)
    else:
        return JupyterBackend(port=port, **kwargs)


def determine_backend(force_webaudio=False, port=8765, **kwargs) -> "BackendBase":
    """Determine a suitable Backend implementation

    This will first try a local pyaudio Backend unless force_webaudio is set.

    Parameters
    ----------
    force_webaudio : bool, optional
        prefer JupyterBackend, by default False
    port : int, optional
        port for the JupyterBackend, by default 8765

    Returns
    -------
    PyAudioBackend or JupyterBackend
        Backend instance

    Raises
    ------
    RuntimeError
        if no Backend is available
    """
    backend = None if force_webaudio else try_pyaudio_backend(**kwargs)
    if backend is None:
        backend = try_jupyter_backend(port=port, **kwargs)
    if backend is None:
        raise RuntimeError(
            "Could not find a backend. "
            "To use the Aserver install the 'pyaudio' or 'remote' extra."
        )
    return backend
