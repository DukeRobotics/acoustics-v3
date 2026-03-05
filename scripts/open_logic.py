import subprocess
import time
import os
import stat

appimage_path = "../Logic-1.2.40-Linux.AppImage"

def launch_logic(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    # Ensure it is executable
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

    # xvfb-run creates the virtual display
    # -a auto-selects a free display number
    # --no-sandbox is usually required for headless AppImages
    # cmd = ["xvfb-run", "-a", path, "--no-sandbox"]
    cmd = ["xvfb-run", "-a", path]

    print("--- Launching Logic Headlessly ---")
    print("Press Ctrl+C to stop the process.")

    process = None
    try:
        process = subprocess.Popen(cmd)
        
        while process.poll() is None:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n--- Stopping Logic... ---")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Logic has been stopped.")

if __name__ == "__main__":
    launch_logic(appimage_path)
