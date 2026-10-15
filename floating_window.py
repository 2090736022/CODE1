# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk


WINDOW_WIDTH = 360
WINDOW_HEIGHT = 100
MARGIN_LEFT = 16
MARGIN_BOTTOM = 72


def place_bottom_left(root: tk.Tk) -> None:
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = MARGIN_LEFT
    y = max(0, screen_height - WINDOW_HEIGHT - MARGIN_BOTTOM)
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")


def enforce_topmost(root: tk.Tk, enabled_var: tk.BooleanVar) -> None:
    if enabled_var.get():
        root.attributes("-topmost", True)
        root.lift()
    root.after(1000, enforce_topmost, root, enabled_var)


def main() -> None:
    root = tk.Tk()
    root.title("悬浮小窗")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.attributes("-toolwindow", True)

    topmost_var = tk.BooleanVar(value=True)

    container = ttk.Frame(root, padding=10)
    container.pack(fill="both", expand=True)

    title = ttk.Label(container, text="悬浮小窗")
    title.pack(anchor="w")

    status = ttk.Label(container, text="始终置顶，位于左下角")
    status.pack(anchor="w", pady=(8, 8))

    controls = ttk.Frame(container)
    controls.pack(fill="x", side="bottom")

    def toggle_topmost() -> None:
        topmost_var.set(not topmost_var.get())
        root.attributes("-topmost", topmost_var.get())
        if topmost_var.get():
            root.lift()

    pin_text = tk.StringVar(value="取消置顶")

    def refresh_pin_text() -> None:
        pin_text.set("取消置顶" if topmost_var.get() else "置顶")

    def on_toggle() -> None:
        toggle_topmost()
        refresh_pin_text()

    pin_button = ttk.Button(controls, textvariable=pin_text, command=on_toggle)
    pin_button.pack(side="left")

    close_button = ttk.Button(controls, text="关闭", command=root.destroy)
    close_button.pack(side="right")

    place_bottom_left(root)
    refresh_pin_text()
    enforce_topmost(root, topmost_var)

    root.mainloop()


if __name__ == "__main__":
    main()
