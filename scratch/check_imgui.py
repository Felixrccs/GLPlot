import imgui
print(f"ImGui version: {imgui.__version__}")
try:
    print(f"Has Docking: {hasattr(imgui, 'dock_space')}")
except Exception as e:
    print(f"Error checking docking: {e}")
