from __future__ import annotations
from pathlib import Path
import faulthandler
import os
import subprocess
import sys
import traceback


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = (
    VENV_DIR / "Scripts" / "python.exe"
    if os.name == "nt"
    else VENV_DIR / "bin" / "python"
)


def _install_requirements():
    requirements = PROJECT_DIR / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError("Cannot find requirements.txt next to app.py.")
    subprocess.run(
        [
            str(VENV_PYTHON), "-m", "pip", "install",
            "--disable-pip-version-check", "-r", str(requirements),
        ],
        check=True,
    )


def _dependencies_available():
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", "import cv2, numpy, PIL, tkinter"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _start_with_local_python():
    if not VENV_PYTHON.is_file():
        print("Creating the local Python environment.", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    if not _dependencies_available():
        print("Installing the required Python packages.", flush=True)
        _install_requirements()
    if Path(sys.prefix).resolve() == VENV_DIR.resolve():
        return None
    print("Starting the puzzle with the local Python environment.", flush=True)
    completed = subprocess.run(
        [str(VENV_PYTHON), str(PROJECT_DIR / "app.py"), *sys.argv[1:]],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    _startup_log = (PROJECT_DIR / "startup.log").open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_startup_log)
    print(f"Starting with Python {sys.version.split()[0]}.", file=_startup_log)

    def _log_exception(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=_startup_log)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_exception
    try:
        _child_exit_code = _start_with_local_python()
    except BaseException:
        traceback.print_exc(file=_startup_log)
        raise
    if _child_exit_code is not None:
        print(f"Game process exited with status {_child_exit_code}.", file=_startup_log)
        raise SystemExit(_child_exit_code)

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from models import ImageProcessor, PuzzleBoard


class Application(tk.Frame):
    """The Tkinter window for playing the image puzzle."""

    MAX_HINTS = 3
    SHIFT_MASK = 0x0001
    CONTROL_MASK = 0x0004

    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Image Tile Puzzle")
        self.master.configure(bg="#f3f6fa")

        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        self._max_image_side = max(
            5, min(680, (screen_width - 170) // 2, screen_height - 310)
        )
        window_width = min(screen_width - 50, 2 * self._max_image_side + 120)
        window_height = min(screen_height - 70, self._max_image_side + 270)
        self.master.geometry(f"{window_width}x{window_height}")
        self.master.minsize(min(670, window_width), min(460, window_height))
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.board = None
        self._selected_index = None
        self._hint_marks = []
        self._hints_used = 0
        self._game_over = False
        self._original_photo = None
        self._puzzle_photo = None

        self.grid(row=0, column=0, sticky="nsew")
        self.create_widgets()
        self._show_placeholders()
        self.refresh()

    def create_widgets(self):
        self.configure(bg="#f3f6fa", padx=18, pady=14)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        heading = tk.Label(
            self, text="Image Tile Puzzle", bg="#f3f6fa",
            font=("TkDefaultFont", 20, "bold"), anchor="w"
        )
        heading.grid(row=0, column=0, sticky="ew")

        controls = tk.Frame(self, bg="#f3f6fa")
        controls.grid(row=1, column=0, sticky="w", pady=(12, 0))
        tk.Label(controls, text="Grid size:", bg="#f3f6fa").grid(
            row=0, column=0, padx=(0, 6)
        )
        self.grid_size_var = tk.StringVar(master=self.master, value="3 × 3")
        self.grid_choice = tk.OptionMenu(
            controls, self.grid_size_var, "3 × 3", "4 × 4", "5 × 5",
            command=self._on_grid_choice_changed
        )
        self.grid_choice.grid(row=0, column=1, padx=(0, 12))
        self.load_button = tk.Button(
            controls, text="Load image", command=self._load_image
        )
        self.load_button.grid(row=0, column=2, padx=(0, 8))
        self.hint_button = tk.Button(controls, text="Hint", command=self._show_hint)
        self.hint_button.grid(row=0, column=3, padx=(0, 8))
        self.flip_button = tk.Button(
            controls, text="Flip selected", command=self._flip_selected
        )
        self.flip_button.grid(row=0, column=4, padx=(0, 8))
        self.solve_button = tk.Button(controls, text="Solve", command=self._solve)
        self.solve_button.grid(row=0, column=5)

        scores = tk.Frame(self, bg="#f3f6fa")
        scores.grid(row=2, column=0, sticky="w", pady=(12, 12))
        self.moves_var = tk.StringVar(master=self.master)
        self.incorrect_var = tk.StringVar(master=self.master)
        self.hints_var = tk.StringVar(master=self.master)
        for column, variable in enumerate(
            (self.moves_var, self.incorrect_var, self.hints_var)
        ):
            tk.Label(
                scores, textvariable=variable, bg="#f3f6fa",
                font=("TkDefaultFont", 11, "bold")
            ).grid(row=0, column=column, padx=(0, 22), sticky="w")

        pictures = tk.Frame(self, bg="#f3f6fa")
        pictures.grid(row=3, column=0, sticky="nsew")
        pictures.columnconfigure(0, weight=1)
        pictures.columnconfigure(1, weight=1)

        original_panel = tk.Frame(pictures, bg="#f3f6fa")
        original_panel.grid(row=0, column=0, sticky="n", padx=(0, 10))
        tk.Label(
            original_panel, text="Original image", bg="#f3f6fa",
            font=("TkDefaultFont", 11, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.original_canvas = tk.Canvas(
            original_panel, bg="#e8edf3", highlightthickness=1,
            highlightbackground="#cbd5e1"
        )
        self.original_canvas.grid(row=1, column=0)

        puzzle_panel = tk.Frame(pictures, bg="#f3f6fa")
        puzzle_panel.grid(row=0, column=1, sticky="n", padx=(10, 0))
        tk.Label(
            puzzle_panel, text="Puzzle", bg="#f3f6fa",
            font=("TkDefaultFont", 11, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.puzzle_canvas = tk.Canvas(
            puzzle_panel, bg="#e8edf3", cursor="hand2",
            highlightthickness=1, highlightbackground="#cbd5e1"
        )
        self.puzzle_canvas.grid(row=1, column=0)
        self.puzzle_canvas.bind("<Button-1>", self._on_left_click)
        self.puzzle_canvas.bind("<Button-2>", self._on_right_click)
        self.puzzle_canvas.bind("<Button-3>", self._on_right_click)

        instructions = (
            "Left click: select or swap  •  Right click: rotate clockwise  •  "
            "Shift + left click, or Flip selected: flip horizontally  •  "
            "Green ticks: correct tiles"
        )
        tk.Label(
            self, text=instructions, bg="#f3f6fa", anchor="w", justify="left",
            wraplength=2 * self._max_image_side + 70
        ).grid(row=4, column=0, sticky="ew", pady=(12, 4))
        self.status_var = tk.StringVar(
            master=self.master,
            value="Choose a grid size, then load a JPG, PNG or BMP image."
        )
        tk.Label(
            self, textvariable=self.status_var, bg="#f3f6fa", anchor="w",
            justify="left", wraplength=2 * self._max_image_side + 70
        ).grid(row=5, column=0, sticky="ew")

    def _show_placeholders(self):
        side = min(self._max_image_side, 420)
        for canvas, label in (
            (self.original_canvas, "Original image"),
            (self.puzzle_canvas, "Scrambled puzzle")
        ):
            canvas.config(width=side, height=side)
            canvas.create_text(
                side // 2, side // 2, text=label, fill="#64748b",
                font=("TkDefaultFont", 13)
            )

    def _load_image(self):
        path = filedialog.askopenfilename(
            parent=self.master,
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Bitmap", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        grid_size = int(self.grid_size_var.get().split()[0])
        try:
            image = ImageProcessor.load_and_prepare(
                path, grid_size, self._max_image_side
            )
            board = PuzzleBoard(image, grid_size)
        except Exception as error:
            messagebox.showerror("Could not load image", str(error), parent=self.master)
            return

        self.board = board
        self._selected_index = None
        self._hint_marks.clear()
        self._hints_used = 0
        self._game_over = False
        self.refresh()
        self.status_var.set(f"Loaded {Path(path).name}. Restore the puzzle on the right.")

    def _tile_at(self, x, y):
        if self.board is None or self._game_over:
            return None
        side = self.board.tile_size * self.board.grid_size
        if not (0 <= x < side and 0 <= y < side):
            return None
        column = x // self.board.tile_size
        row = y // self.board.tile_size
        return row * self.board.grid_size + column

    def _draw_images(self):
        if self.board is None:
            return
        original = self.board.original_image
        scrambled = self.board.render_image()
        height, width = original.shape[:2]

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        scrambled_rgb = cv2.cvtColor(scrambled, cv2.COLOR_BGR2RGB)
        self._original_photo = ImageTk.PhotoImage(Image.fromarray(original_rgb))
        self._puzzle_photo = ImageTk.PhotoImage(Image.fromarray(scrambled_rgb))
        for canvas in (self.original_canvas, self.puzzle_canvas):
            canvas.config(width=width, height=height)
            canvas.delete("all")
        self.original_canvas.create_image(
            0, 0, anchor="nw", image=self._original_photo
        )
        self.puzzle_canvas.create_image(0, 0, anchor="nw", image=self._puzzle_photo)

        tile_size = self.board.tile_size
        for line in range(1, self.board.grid_size):
            position = line * tile_size
            self.puzzle_canvas.create_line(
                position, 0, position, height, fill="#aab6c3"
            )
            self.puzzle_canvas.create_line(
                0, position, width, position, fill="#aab6c3"
            )

        for index in range(self.board.grid_size ** 2):
            if self.board.tile_is_correct(index):
                self._draw_tick(index)

        if self._selected_index is not None:
            row, column = divmod(self._selected_index, self.board.grid_size)
            left = column * tile_size
            top = row * tile_size
            self.puzzle_canvas.create_rectangle(
                left + 2, top + 2, left + tile_size - 2, top + tile_size - 2,
                outline="#f97316", width=4
            )

        for current, home in self._hint_marks:
            self._draw_hint_circle(self.puzzle_canvas, current)
            self._draw_hint_circle(self.original_canvas, home)

        if self._game_over and self.board.is_solved:
            # Keep a visible completion notice after the dialog is closed.
            banner_height = min(46, max(24, height // 8))
            self.puzzle_canvas.create_rectangle(
                0, height - banner_height, width, height,
                fill="#166534", outline="#166534"
            )
            self.puzzle_canvas.create_text(
                width // 2, height - banner_height // 2,
                text="Puzzle complete! Load another image",
                fill="white", font=("TkDefaultFont", 12, "bold")
            )

    def _draw_tick(self, index):
        tile_size = self.board.tile_size
        row, column = divmod(index, self.board.grid_size)
        size = max(8, min(17, tile_size // 5))
        left = column * tile_size + tile_size - size - 7
        top = row * tile_size + 8
        points = (
            left, top + size * 0.48,
            left + size * 0.35, top + size * 0.78,
            left + size, top
        )
        self.puzzle_canvas.create_line(
            *points, fill="white", width=6,
            capstyle="round", joinstyle="round"
        )
        self.puzzle_canvas.create_line(
            *points, fill="#16a34a", width=3,
            capstyle="round", joinstyle="round"
        )

    def _draw_hint_circle(self, canvas, index):
        tile_size = self.board.tile_size
        row, column = divmod(index, self.board.grid_size)
        x = (column + 0.5) * tile_size
        y = (row + 0.5) * tile_size
        radius = max(10, min(22, tile_size // 5))
        canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            outline="white", width=7
        )
        canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            outline="#2563eb", width=4
        )

    def _on_left_click(self, event):
        index = self._tile_at(event.x, event.y)
        if index is None:
            return
        if event.state & self.SHIFT_MASK:
            self._make_move(lambda: self.board.flip_horizontal(index))
            return
        if event.state & self.CONTROL_MASK:
            self._make_move(lambda: self.board.rotate(index))
            return

        if self._selected_index is None:
            self._selected_index = index
            self._draw_images()
            self.status_var.set(
                "Tile selected. Click another tile to swap, or press Flip selected."
            )
        elif self._selected_index == index:
            self._selected_index = None
            self._draw_images()
        else:
            first = self._selected_index
            self._make_move(lambda: self.board.swap(first, index))

    def _on_right_click(self, event):
        index = self._tile_at(event.x, event.y)
        if index is not None:
            self._make_move(lambda: self.board.rotate(index))

    def _flip_selected(self):
        """Flip the highlighted tile with a button as well as Shift-click."""
        if self.board is None or self._game_over:
            return
        if self._selected_index is None:
            self.status_var.set(
                "Left-click a puzzle tile to highlight it, then press Flip selected."
            )
            return
        self._make_move(lambda: self.board.flip_horizontal(self._selected_index))

    def _make_move(self, action):
        try:
            accepted = action()
        except Exception as error:
            messagebox.showerror(
                "Move could not be completed", str(error), parent=self.master
            )
            return
        if not accepted:
            return
        self._selected_index = None
        self._hint_marks.clear()
        if self.board.is_solved:
            self._game_over = True
            self.refresh()
            self.status_var.set("Picture restored! Load another image to play again.")
            if self.master is not None:
                self.master.update_idletasks()
            messagebox.showinfo(
                "Puzzle complete",
                f"You restored the picture in {self.board.move_count} moves!",
                parent=self.master
            )
        else:
            self.refresh()
            self.status_var.set(
                f"{self.board.incorrect_count} tiles still need the right "
                "position or orientation. Green ticks mark finished tiles."
            )

    def _show_hint(self):
        if self.board is None or self._game_over or self._hints_used >= self.MAX_HINTS:
            return
        hint = self.board.get_hint()
        if hint is None:
            return
        self._hint_marks[:] = [hint]
        self._hints_used += 1
        self.refresh()
        current, home = hint
        if current == home:
            self.status_var.set(
                "The circled tile is in its home square. Try flipping it with "
                "Shift + left click, or rotate it with right click."
            )
        else:
            self.status_var.set(
                "The blue circles show an incorrect tile and its home square. "
                "Swap it into place, then rotate or flip if needed."
            )

    def _on_grid_choice_changed(self, _choice):
        if self.board is None:
            self.status_var.set("Load an image to use this grid size.")
        else:
            self.status_var.set("This grid size will apply to the next image you load.")

    def _solve(self):
        if self.board is None or self._game_over:
            return
        try:
            self.board.solve()
        except Exception as error:
            messagebox.showerror("Could not solve puzzle", str(error), parent=self.master)
            return
        self._selected_index = None
        self._hint_marks.clear()
        self._game_over = True
        self.refresh()
        self.status_var.set(
            "Puzzle solved. Moves and incorrect tiles are reset; "
            "load another image to play."
        )
        if self.master is not None:
            self.master.update_idletasks()
        messagebox.showinfo(
            "Puzzle solved", "The picture has been restored. Load another image to play.",
            parent=self.master
        )

    def refresh(self):
        """Update the picture, score and buttons after the puzzle changes."""
        if self.board is not None:
            self._draw_images()
        self._update_stats()
        self._refresh_controls()

    def _update_stats(self):
        if self.board is None:
            self.moves_var.set("Moves: 0")
            self.incorrect_var.set("Tiles incorrect: —")
        else:
            self.moves_var.set(f"Moves: {self.board.move_count}")
            self.incorrect_var.set(
                f"Tiles incorrect: {self.board.incorrect_count}"
            )
        self.hints_var.set(f"Hints left: {self.MAX_HINTS - self._hints_used}")

    def _refresh_controls(self):
        playing = self.board is not None and not self._game_over
        self.solve_button.config(state="normal" if playing else "disabled")
        self.flip_button.config(state="normal" if playing else "disabled")
        self.hint_button.config(
            state="normal" if playing and self._hints_used < self.MAX_HINTS
            else "disabled"
        )


def main():
    if "--check" in sys.argv[1:]:
        print("Dependencies are ready.")
        return
    root = tk.Tk()
    Application(master=root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    finally:
        _startup_log.close()
