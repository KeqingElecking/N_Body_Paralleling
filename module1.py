import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
src_name = "kernel.cu"
src_path = os.path.join(script_dir, src_name)
exe_name = "nbody.exe" if os.name == "nt" else "nbody"
exe_path = os.path.join(script_dir, exe_name)

if not os.path.isfile(src_path):
    sys.stderr.write(f"Source '{src_name}' not found in {script_dir}\n")
    sys.exit(1)

def find_executable(candidates=None, root=script_dir):
    """
    Search for executables in the workspace.
    - candidates: list of preferred filenames (checked first).
    - root: directory to walk (script_dir by default).
    Returns full path to first match or None.
    """
    # Check exact candidate names in script_dir and PATH first
    if candidates:
        for name in candidates:
            # local candidate in script dir and subdirs
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if os.name == "nt":
                        if f.lower() == name.lower():
                            return os.path.join(dirpath, f)
                    else:
                        if f == name:
                            full = os.path.join(dirpath, f)
                            if os.access(full, os.X_OK):
                                return full
            # check PATH
            which = shutil.which(name)
            if which:
                return which

    # If nothing matched, look for any .exe (Windows) or any executable file (POSIX)
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            if os.name == "nt":
                if f.lower().endswith(".exe"):
                    return full
            else:
                if os.access(full, os.X_OK):
                    return full
    return None

# Prefer these names in order: compiled binary (nbody), then CudaRuntime1
preferred = [exe_name, "CudaRuntime1.exe" if os.name == "nt" else "CudaRuntime1"]
found = find_executable(preferred)

if found:
    print(f"Found executable: {found}")
    # If it's a CudaRuntime runtime, pass the .cu source; otherwise run the executable directly.
    basename = os.path.basename(found).lower()
    if "cudaruntime" in basename:
        run_cmd = [found, src_path]
    else:
        run_cmd = [found]
else:
    print("No existing executable found; will attempt to compile locally.")
    # Choose compiler: prefer nvcc, then g++, then cl (Windows)
    compiler = None
    compile_cmd = None
    if shutil.which("nvcc"):
        compiler = "nvcc"
        if os.name == "nt" and not shutil.which("cl"):
            host_cc = shutil.which("g++")
            if host_cc:
                print("Warning: cl.exe not found. Trying nvcc with -ccbin pointing to g++ (may fail).")
                compile_cmd = ["nvcc", "-ccbin", host_cc, "-std=c++14", src_path, "-O3", "-o", exe_path]
            else:
                sys.stderr.write("nvcc requires MSVC (cl.exe) on Windows and cl.exe was not found in PATH.\n")
                sys.stderr.write("Open a __Developer Command Prompt__ or install MSVC Build Tools.\n")
                sys.exit(1)
        else:
            compile_cmd = ["nvcc", "-std=c++14", src_path, "-O3", "-o", exe_path]
    elif shutil.which("g++"):
        compiler = "g++"
        compile_cmd = ["g++", "-std=c++14", "-O3", src_path, "-o", exe_path]
    elif os.name == "nt" and shutil.which("cl"):
        compiler = "cl"
        compile_cmd = ["cl", "/EHsc", "/O2", src_path, f"/Fe:{exe_path}"]
    else:
        sys.stderr.write("No suitable compiler found (nvcc or g++). Install CUDA (nvcc) or g++ and add to PATH.\n")
        sys.exit(1)

    print(f"Compiling {src_name} with {compiler} ...")
    proc = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write("Compilation failed.\n")
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        if "cl.exe" in (proc.stderr or ""):
            sys.stderr.write("\nOn Windows nvcc needs MSVC (cl.exe) on PATH. Start a __Developer Command Prompt__ or run vcvars.\n")
        sys.exit(proc.returncode)
    found = exe_path
    run_cmd = [found]

# Run and capture output
print("Running:", " ".join(run_cmd))
run_proc = subprocess.run(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if run_proc.returncode != 0:
    sys.stderr.write(f"'{ ' '.join(run_cmd) }' exited with code {run_proc.returncode}\n")
    sys.stderr.write(run_proc.stderr or "")
    sys.exit(run_proc.returncode)

output = run_proc.stdout

# Parse output into frames (3D: x y z per line)
frames = []
current_frame = []

for raw_line in output.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    if line.lower() == "timestep":
        if current_frame:
            frames.append(np.asarray(current_frame, dtype=float))
        current_frame = []
        continue
    parts = line.split()
    if len(parts) < 3:
        continue
    try:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        current_frame.append([x, y, z])
    except ValueError:
        continue

if current_frame:
    frames.append(np.asarray(current_frame, dtype=float))

if not frames:
    sys.stderr.write("No frames parsed from executable output.\n")
    sys.exit(1)

# Prepare 3D animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("black")

# concatenate to compute bounds
all_points = np.vstack(frames)
mins = np.min(all_points, axis=0) - 0.1
maxs = np.max(all_points, axis=0) + 0.1

# make symmetric bounds around origin for better viewing
xpad = max(abs(mins[0]), abs(maxs[0]))
ypad = max(abs(mins[1]), abs(maxs[1]))
zpad = max(abs(mins[2]), abs(maxs[2]))
ax.set_xlim(-xpad, xpad)
ax.set_ylim(-ypad, ypad)
ax.set_zlim(-zpad, zpad)

first = frames[0]
scatter = ax.scatter(first[:, 0], first[:, 1], first[:, 2], s=10, c="cyan", alpha=0.8)

def update(frame_idx):
    data = frames[frame_idx]
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    # update 3D scatter
    scatter._offsets3d = (xs, ys, zs)
    return (scatter,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=False)

print("Rendering animation...")
plt.show()
