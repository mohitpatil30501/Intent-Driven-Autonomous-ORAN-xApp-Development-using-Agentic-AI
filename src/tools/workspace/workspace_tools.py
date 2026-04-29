import os
import subprocess
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import tool

# Ensure workspace directory exists
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "workspace"))
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 1. File System Tools
file_toolkit = FileManagementToolkit(
    root_dir=WORKSPACE_DIR,
    selected_tools=["read_file", "write_file", "list_directory", "copy_file", "move_file", "file_search"]
)
file_tools = file_toolkit.get_tools()

# 2. Restricted Terminal Tool
@tool
def terminal_command(command: str) -> str:
    """
    Execute a shell command within the restricted workspace directory.
    Use this to run compilations, python scripts, or other shell utilities.
    The command will automatically be executed with the workspace directory as its current working directory.
    """
    # Security: rudimentary block of traversing out of workspace
    if "cd .." in command or "cd /" in command or "cd ~" in command:
        return "Error: You are not allowed to change directories outside the workspace."
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output if output else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

# Combine all workspace tools
workspace_tools = file_tools + [terminal_command]
