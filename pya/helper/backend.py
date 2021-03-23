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


def determine_backend(force_webaudio=False, port=8765):
    import os
    server_info = get_server_info()
    if server_info is None or (server_info['hostname'] in ['localhost', '127.0.0.1'] and not force_webaudio):
        return None  # use default local backend
    from ..backend.Jupyter import JupyterBackend
    if os.environ.get('BINDER_SERVICE_HOST'):
        return JupyterBackend(port=port, proxy_suffix=f"/proxy/{port}")
    else:
        return JupyterBackend(port=port)
