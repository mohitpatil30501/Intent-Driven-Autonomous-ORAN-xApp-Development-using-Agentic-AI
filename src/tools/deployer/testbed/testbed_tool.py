import os
import subprocess
import time
from langchain_core.tools import tool

# Base directory for the testbed environment
TESTBED_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NEARRTRIC_DIR = os.path.join(TESTBED_BASE_DIR, "nearrtric")
XAPPS_DIR = os.path.join(NEARRTRIC_DIR, "xapps")

def ensure_testbed_running():
    """
    Ensures that the testbed (RIC, RAN, etc.) is up and running.
    If not running, it starts it using the start.sh script.
    Also checks if the RIC is healthy.
    """
    # 1. Check if core components are running
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=mysql", "--format", "{{.Status}}"],
        capture_output=True,
        text=True
    )
    
    if not result.stdout or "Up" not in result.stdout:
        print("Testbed components not running. Starting via start.sh...")
        start_script = os.path.join(TESTBED_BASE_DIR, "start.sh")
        if os.path.exists(start_script):
            subprocess.run(["bash", start_script], cwd=TESTBED_BASE_DIR)
            time.sleep(30)
        else:
            print(f"Error: start.sh not found at {start_script}")
            return

    # 2. Check if FlexRIC is healthy
    ric_check = subprocess.run(
        ["docker", "ps", "--filter", "name=flexric", "--format", "{{.Status}}"],
        capture_output=True,
        text=True
    )
    
    if "unhealthy" in ric_check.stdout.lower() or "exited" in ric_check.stdout.lower():
        print("FlexRIC is unhealthy. Restarting nearrtric components...")
        nearrtric_dir = os.path.join(TESTBED_BASE_DIR, "nearrtric")
        subprocess.run(["docker", "compose", "restart"], cwd=nearrtric_dir)
        time.sleep(15)
    
    print("Testbed is running and components checked.")

import base64

@tool
def deploy_xapp_to_testbed(xapp_code: str, artifacts: dict = None, timeout: int = 15) -> str:
    """
    Deploys and executes an xApp in the FlexRIC testbed environment.
    Optimized to update code and restart the xapp container.
    
    Args:
        xapp_code (str): The complete Python code for the xApp.
        artifacts (dict): Optional dictionary mapping filenames to content (str or base64).
                          E.g. {"model.pkl": "base64_data..."}
        timeout (int): Seconds to wait for the xApp to run before capturing logs.
        
    Returns:
        str: The STDOUT and STDERR logs from the xApp execution.
    """
    if not os.path.exists(NEARRTRIC_DIR):
        return f"Error: Testbed environment not found at {NEARRTRIC_DIR}."

    try:
        # 1. Ensure testbed is running
        ensure_testbed_running()

        # 2. Use a hidden file in the actual XAPPS_DIR for staging
        # This keeps it in the same context as dependencies if needed
        staging_xapp_path = os.path.join(XAPPS_DIR, ".tmp_xapp_deploy.py")
        
        # Write the xapp_code to the staging file
        with open(staging_xapp_path, "w") as f:
            f.write(xapp_code)

        # Handle additional artifacts
        artifact_staging_paths = []
        if artifacts:
            for filename, content in artifacts.items():
                target_path = os.path.join(XAPPS_DIR, f".tmp_{filename}")
                try:
                    if len(content) > 100 and " " not in content:
                        decoded_data = base64.b64decode(content)
                        with open(target_path, "wb") as f:
                            f.write(decoded_data)
                    else:
                        with open(target_path, "w") as f:
                            f.write(content)
                except Exception:
                    with open(target_path, "w") as f:
                        f.write(content)
                artifact_staging_paths.append((target_path, filename))

        # 3. Copy everything into the running container
        container_base = "/flexric/build/examples/xApp/python3/"
        
        # Copy the staged xApp code as 'xapp.py' in the container
        subprocess.run(
            ["docker", "cp", staging_xapp_path, f"xapp:{container_base}xapp.py"],
            check=True
        )

        # Copy other artifacts to their correct names in the container
        for host_path, container_name in artifact_staging_paths:
            subprocess.run(
                ["docker", "cp", host_path, f"xapp:{container_base}{container_name}"],
                check=True
            )

        # 4. Clean up staging files from the host
        if os.path.exists(staging_xapp_path):
            os.remove(staging_xapp_path)
        for host_path, _ in artifact_staging_paths:
            if os.path.exists(host_path):
                os.remove(host_path)

        # 5. Restart only the xapp container
        subprocess.run(["docker", "restart", "xapp"], check=True)

        # 6. Wait for requested timeout for the xApp to run
        print(f"Waiting {timeout}s for xApp to run...")
        time.sleep(timeout)

        # 7. Capture logs from the xapp container
        logs_result = subprocess.run(
            ["docker", "logs", "--tail", "100", "xapp"],
            capture_output=True,
            text=True
        )
        
        output = logs_result.stdout
        if logs_result.stderr:
            output += f"\nSTDERR:\n{logs_result.stderr}"

        return output if output else "xApp executed, but no logs were captured."

    except Exception as e:
        return f"Error during xApp deployment: {str(e)}"
