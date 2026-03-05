import subprocess
import os
import stat

appimage_path = "../Logic-1.2.40-Linux.AppImage"

def launch_logic(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

    print(f"Launching {path}...")
    try:
        # subprocess.Popen([path, "--no-sandbox"])
        subprocess.Popen([path])
    except Exception as e:
        print(f"Failed to launch: {e}")

if __name__ == "__main__":
    launch_logic(appimage_path)
