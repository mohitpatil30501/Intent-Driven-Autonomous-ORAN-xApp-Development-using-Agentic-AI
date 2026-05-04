import os
import shutil
import subprocess
import time
from langchain_core.messages import AIMessage

def module_7_deployer_node(state: dict) -> dict:
    """Module 7: Copies artifacts to testbed and runs it for 20 seconds."""
    blueprint = state.get("blueprint", {})
    messages = state.get("messages", [])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_dir = os.path.join(base_dir, "workspace")
    testbed_dir = os.path.join(workspace_dir, "testbed")
    nearrtric_dir = os.path.join(testbed_dir, "nearrtric")
    xapps_dir = os.path.join(nearrtric_dir, "xapps")

    # 1. Copy final_xapp.py
    final_xapp_src = os.path.join(workspace_dir, "final_xapp.py")
    if os.path.exists(final_xapp_src):
        shutil.copy(final_xapp_src, os.path.join(xapps_dir, "xapp.py"))
    else:
        print("Warning: final_xapp.py not found in workspace.")
    
    # 2. Copy directories (data, ml, logic)
    for folder in ["data", "ml", "logic"]:
        src_folder = os.path.join(workspace_dir, folder)
        dest_folder = os.path.join(xapps_dir, folder)
        if os.path.exists(src_folder):
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
            shutil.copytree(src_folder, dest_folder)

    # 3. Update Dockerfile.xapp
    dockerfile_path = os.path.join(nearrtric_dir, "Dockerfile.xapp")
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path, "r") as f:
            dockerfile_content = f.read()
        
        if "COPY ./xapps/xapp.py /flexric/build/examples/xApp/python3/xapp.py" in dockerfile_content:
            dockerfile_content = dockerfile_content.replace(
                "COPY ./xapps/xapp.py /flexric/build/examples/xApp/python3/xapp.py",
                "COPY ./xapps /flexric/build/examples/xApp/python3/"
            )
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

    # 4. Stop existing containers
    subprocess.run(["docker", "compose", "down"], cwd=nearrtric_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 5. Start nearrtric with build
    print("Building and starting xApp in testbed...")
    subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=nearrtric_dir)
    
    # Wait for 20 seconds
    print("Running for 20 seconds...")
    time.sleep(20)
    
    # 6. Get logs and summarize
    result = subprocess.run(["docker", "logs", "--tail", "50", "xapp"], cwd=nearrtric_dir, capture_output=True, text=True)
    logs = result.stdout + result.stderr
    
    # Stop containers after test
    subprocess.run(["docker", "compose", "down"], cwd=nearrtric_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    summary = f"Module 7: Deployment completed. Ran for 20 seconds.\n\nxApp Logs Summary:\n{logs[-1000:]}"
    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=summary)],
        "is_complete": True
    }
