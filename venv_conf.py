import os
import shutil

def move_and_delete_venv():
    root_folder = os.getcwd()
    default_venv_path = os.path.join(root_folder, 'default.venv')
    target_venv_path = os.path.join(root_folder, '.venv')

    # Check if default.venv exists
    if os.path.exists(default_venv_path):
        # Check if .venv folder exists inside default.venv
        venv_inside_default_path = os.path.join(default_venv_path, '.venv')
        if os.path.exists(venv_inside_default_path) and os.path.isdir(venv_inside_default_path):
            # Move .venv folder to root
            shutil.move(venv_inside_default_path, target_venv_path)
            # Delete default.venv folder
            os.rmdir(default_venv_path)
            print("Moved .venv folder to root and deleted default.venv folder.")
        else:
            print("No .venv folder found inside default.venv.")
    else:
        print("default.venv folder not found in the root directory.")

if __name__ == "__main__":
    move_and_delete_venv()
