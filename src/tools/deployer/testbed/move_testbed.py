import os
import shutil

src = "/home/mohit/MTech/AutORAN/src/workspace/testbed"
dst = "/home/mohit/MTech/AutORAN/src/tools/deployer/testbed/env"

if os.path.exists(src):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print(f"Moved {src} to {dst}")
else:
    print(f"Source {src} not found")
