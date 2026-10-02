"""HIT137 Assignment 3: Picture Puzzle.

Group name: Group-30
Group members:
William Wendl - S361240
Kamala Thapa - S407760
Usha Khadka - S408025

Run: python image_puzzle.py
Install: python -m pip install -r requirements.txt
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as error:
    raise SystemExit(
        "A required package is missing. In this folder, run:\n"
        "python -m pip install -r requirements.txt\n"
        f"Details: {error}"
    ) from error


class ImageLoader:
    """Load through OpenCV and preserve aspect ratio inside a square board."""

    @staticmethod
    def load(path: str | Path, grid_size: int, max_side: int) -> np.ndarray:
        # Reading bytes first also supports non-ASCII filenames on Windows.
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        if data.size == 0:
            raise ValueError("This file is empty. Choose a JPG, PNG or BMP image.")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("This file could not be read as an image.")
        return ImageLoader.prepare(image, grid_size, max_side)

    @staticmethod
    def prepare(image: np.ndarray, grid_size: int, max_side: int) -> np.ndarray:
        if grid_size not in (3, 4, 5):
            raise ValueError("The grid size must be 3, 4 or 5.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("The image must contain three colour channels.")
        if max_side < grid_size:
            raise ValueError("The display is too small for the chosen grid.")

        # A square board makes 90-degree rotations possible without stretching.
        side = (int(max_side) // grid_size) * grid_size
        height, width = image.shape[:2]
        scale = side / max(height, width)
        new_width = max(1, min(side, round(width * scale)))
        new_height = max(1, min(side, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height),
                             interpolation=interpolation)

        left = (side - new_width) // 2
        top = (side - new_height) // 2
        return cv2.copyMakeBorder(
            resized, top, side - new_height - top,
            left, side - new_width - left,
            cv2.BORDER_CONSTANT, value=(232, 237, 242)
        )


class Tile:
    """Encapsulate a tile's identity, pixels and rotation/reflection state."""

    def __init__(self, home_index: int, pixels: np.ndarray):
        self._home_index = home_index
        self._pixels = pixels.copy()
        self._quarter_turns = 0
        self._flipped = False

    @property
    def home_index(self) -> int:
        return self._home_index

    @property
    def pixels(self) -> np.ndarray:
        # Return a copy so callers cannot accidentally edit our stored pixels.
        return self._pixels.copy()

    @property
    def orientation(self) -> tuple[int, bool]:
        return self._quarter_turns, self._flipped

    def is_correct(self, position: int) -> bool:
        return self._home_index == position and self.orientation == (0, False)

    def rotate(self, quarter_turns: int = 1) -> None:
        turns = quarter_turns % 4
        codes = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
                 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if turns:
            self._pixels = cv2.rotate(self._pixels, codes[turns])
            self._quarter_turns = (self._quarter_turns + turns) % 4

    def flip(self, axis: str = "horizontal") -> None:
        if axis not in ("horizontal", "vertical"):
            raise ValueError("Flip axis must be horizontal or vertical.")
        self._pixels = cv2.flip(self._pixels, 1 if axis == "horizontal" else 0)
        # State represents R^k H^f. Reflecting changes the sign of k:
        # H R^k = R^-k H; a vertical reflection is R^2 H.
        offset = 0 if axis == "horizontal" else 2
        self._quarter_turns = (offset - self._quarter_turns) % 4
        self._flipped = not self._flipped


class Transformation(ABC):
    """Common interface for reversible actions (inheritance/polymorphism)."""

    @property
    @abstractmethod
    def targets(self) -> tuple[int, ...]:
        """Board positions affected by this action."""

    @abstractmethod
    def apply(self, puzzle: Puzzle) -> None:
        """Perform this action on a puzzle."""

    @abstractmethod
    def inverse(self) -> Transformation:
        """Return an action that reverses this one."""


class SwapTransformation(Transformation):
    def __init__(self, first: int, second: int):
        if first == second:
            raise ValueError("A swap requires two different tiles.")
        self.first, self.second = first, second

    @property
    def targets(self) -> tuple[int, ...]:
        return self.first, self.second

    def apply(self, puzzle: Puzzle) -> None:
        puzzle._swap_tiles(self.first, self.second)

    def inverse(self) -> Transformation:
        return SwapTransformation(self.first, self.second)


class RotateTransformation(Transformation):
    def __init__(self, index: int, quarter_turns: int = 1):
        if quarter_turns % 4 == 0:
            raise ValueError("A rotation must turn the tile by 90, 180 or 270 degrees.")
        self.index = index
        self.quarter_turns = quarter_turns % 4

    @property
    def targets(self) -> tuple[int, ...]:
        return (self.index,)

    def apply(self, puzzle: Puzzle) -> None:
        puzzle._rotate_tile(self.index, self.quarter_turns)

    def inverse(self) -> Transformation:
        return RotateTransformation(self.index, 4 - self.quarter_turns)


class FlipTransformation(Transformation):
    def __init__(self, index: int, axis: str = "horizontal"):
        if axis not in ("horizontal", "vertical"):
            raise ValueError("Flip axis must be horizontal or vertical.")
        self.index, self.axis = index, axis

    @property
    def targets(self) -> tuple[int, ...]:
        return (self.index,)

    def apply(self, puzzle: Puzzle) -> None:
        puzzle._flip_tile(self.index, self.axis)

    def inverse(self) -> Transformation:
        return FlipTransformation(self.index, self.axis)


class Puzzle:
    """Game model: tile positions, scrambling, moves, hints and solving."""

    TRANSFORMATION_COUNTS = {3: 6, 4: 12, 5: 20}
    MAX_HINTS = 3

    def __init__(self, original: np.ndarray, grid_size: int,
                 rng: random.Random | None = None):
        if grid_size not in self.TRANSFORMATION_COUNTS:
            raise ValueError("The grid size must be 3, 4 or 5.")
        if (original.ndim != 3 or original.shape[2] != 3
                or original.shape[0] == 0
                or original.shape[0] != original.shape[1]
                or original.shape[0] % grid_size):
            raise ValueError("The board must be a square divisible by its grid size.")
        self._original = original.copy()
        self._grid_size = grid_size
        self._tile_size = original.shape[0] // grid_size
        self._rng = rng if rng is not None else random.Random()
        self._tiles = []
        for index in range(grid_size * grid_size):
            row, column = divmod(index, grid_size)
            y, x = row * self._tile_size, column * self._tile_size
            pixels = original[y:y + self._tile_size, x:x + self._tile_size]
            self._tiles.append(Tile(index, pixels))
        self._moves = 0
        self._hints_used = 0
        self._selected: int | None = None
        self._hint: tuple[int, int] | None = None
        self._history: list[Transformation] = []
        self._initial_actions = self._make_scramble()
        # All actions are planned first; no intermediate scramble is displayed.
        for action in self._initial_actions:
            self._apply_and_record(action)

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def side(self) -> int:
        return self._original.shape[0]

    @property
    def original(self) -> np.ndarray:
        return self._original.copy()

    @property
    def moves(self) -> int:
        return self._moves

    @property
    def selected(self) -> int | None:
        return self._selected

    @property
    def hint(self) -> tuple[int, int] | None:
        # (current board position, correct position in the original)
        return self._hint

    @property
    def hints_remaining(self) -> int:
        return self.MAX_HINTS - self._hints_used

    @property
    def incorrect_indices(self) -> tuple[int, ...]:
        return tuple(i for i, tile in enumerate(self._tiles)
                     if not tile.is_correct(i))

    @property
    def tiles_left(self) -> int:
        return len(self.incorrect_indices)

    @property
    def solved(self) -> bool:
        return self.tiles_left == 0

    @property
    def scramble_actions(self) -> tuple[Transformation, ...]:
        return self._initial_actions

    def home_index(self, position: int) -> int:
        self._validate_index(position)
        return self._tiles[position].home_index

    def orientation(self, position: int) -> tuple[int, bool]:
        self._validate_index(position)
        return self._tiles[position].orientation

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or not 0 <= index < len(self._tiles):
            raise IndexError("Tile position is outside the board.")

    def _make_scramble(self) -> tuple[Transformation, ...]:
        positions = list(range(len(self._tiles)))
        self._rng.shuffle(positions)
        count = self.TRANSFORMATION_COUNTS[self._grid_size]
        # A swap consumes two unique positions, but counts as one action.
        # 3/4/5 swaps + 3/8/15 single-tile actions = 6/12/20 actions.
        swap_count = len(positions) - count
        actions: list[Transformation] = []
        for offset in range(0, 2 * swap_count, 2):
            actions.append(SwapTransformation(positions[offset], positions[offset + 1]))

        remaining = positions[2 * swap_count:]
        # Guarantee both single-tile types, then choose the rest randomly.
        kinds = ["rotate", "flip"]
        kinds += [self._rng.choice(("rotate", "flip"))
                  for _ in range(len(remaining) - 2)]
        self._rng.shuffle(kinds)
        for index, kind in zip(remaining, kinds):
            if kind == "rotate":
                actions.append(RotateTransformation(index, self._rng.randint(1, 3)))
            else:
                axis = self._rng.choice(("horizontal", "vertical"))
                actions.append(FlipTransformation(index, axis))
        self._rng.shuffle(actions)
        return tuple(actions)

    def _swap_tiles(self, first: int, second: int) -> None:
        self._validate_index(first)
        self._validate_index(second)
        self._tiles[first], self._tiles[second] = self._tiles[second], self._tiles[first]

    def _rotate_tile(self, index: int, quarter_turns: int) -> None:
        self._validate_index(index)
        self._tiles[index].rotate(quarter_turns)

    def _flip_tile(self, index: int, axis: str) -> None:
        self._validate_index(index)
        self._tiles[index].flip(axis)

    def _apply_and_record(self, action: Transformation) -> None:
        action.apply(self)  # Same interface; the subclass chooses the behaviour.
        self._history.append(action)

    def move(self, action: Transformation) -> bool:
        if self.solved:
            return False
        self._apply_and_record(action)
        self._moves += 1
        self._selected = None
        self._hint = None  # Hints last until the next completed move.
        return True

    def select(self, index: int) -> bool:
        """Return True only if selecting a second tile performs a swap."""
        self._validate_index(index)
        if self.solved:
            return False
        if self._selected is None:
            self._selected = index
            return False
        if self._selected == index:
            self._selected = None
            return False
        return self.move(SwapTransformation(self._selected, index))

    def request_hint(self) -> tuple[int, int] | None:
        if self.solved or self.hints_remaining == 0:
            return None
        position = self._rng.choice(self.incorrect_indices)
        self._hint = position, self.home_index(position)
        self._hints_used += 1
        return self._hint

    def solve(self) -> None:
        # Undo player actions AND the initial scramble in reverse order.
        for action in reversed(self._history):
            action.inverse().apply(self)
        self._history.clear()
        self._moves = 0
        self._selected = None
        self._hint = None
        # Hints remain used until another image is loaded.

    def assemble(self) -> np.ndarray:
        rows = []
        for row in range(self._grid_size):
            start = row * self._grid_size
            rows.append(cv2.hconcat([tile.pixels for tile in
                                    self._tiles[start:start + self._grid_size]]))
        return cv2.vconcat(rows)


class PuzzleApp:
    """Tkinter view/controller; the original canvas has no gameplay bindings."""

    GRID_LABELS = {"3 x 3 - Easy": 3, "4 x 4 - Medium": 4, "5 x 5 - Hard": 5}
    TIME_LIMITS = {3: 300, 4: 480, 5: 600}

    def __init__(self, root: tk.Tk, clock: Callable[[], float] = time.monotonic):
        self.root = root
        self._clock = clock
        self._puzzle: Puzzle | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._started_at = 0.0
        self._finished_at: float | None = None
        self._time_limit: int | None = None
        self._expired = False
        self._timer_id: str | None = None
        self._completion_shown = False

        root.title("HIT137 | Picture Puzzle")
        root.configure(bg="#eef2f6")
        # Leave room for two images, controls, and desktop window decorations.
        self._board_limit = min(440, max(150, (root.winfo_screenwidth() - 130) // 2),
                                max(150, root.winfo_screenheight() - 400))
        self._canvas_size = self._board_limit + 16
        self._board_x = 8
        self._board_y = 8
        self._grid_var = tk.StringVar(value="3 x 3 - Easy")
        self._challenge_var = tk.BooleanVar(value=False)
        self._moves_var = tk.StringVar(value="0")
        self._tiles_var = tk.StringVar(value="-")
        self._hints_var = tk.StringVar(value="3")
        self._time_var = tk.StringVar(value="00:00")
        self._status_var = tk.StringVar(value="Choose a grid, then load an image or try the demo.")
        self._file_var = tk.StringVar(value="No image loaded")
        self._build_widgets()
        self._fit_window_to_screen()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._schedule_clock()

    @property
    def puzzle(self) -> Puzzle | None:
        return self._puzzle

    def _build_widgets(self) -> None:
        available = {family.lower() for family in tkfont.families(self.root)}
        self._font_family = next((family for family in
                                  ("Segoe UI", "DejaVu Sans", "Helvetica", "Arial")
                                  if family.lower() in available),
                                 tkfont.nametofont("TkDefaultFont").actual("family"))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", foreground="#26394b",
                        font=(self._font_family, 10))
        style.configure("TButton", padding=(12, 8), font=(self._font_family, 10))
        style.configure("Accent.TButton", background="#225e76", foreground="white")
        style.map("Accent.TButton", background=[("active", "#184c61")])
        style.configure("TCheckbutton", background="#eef2f6", foreground="#26394b",
                        font=(self._font_family, 10))
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="PICTURE PUZZLE", font=(self._font_family, 23, "bold")).pack(anchor="w")
        ttk.Label(container, text="Swap, turn and flip the pieces to restore the picture.",
                  font=(self._font_family, 11)).pack(anchor="w", pady=(4, 16))

        toolbar = ttk.Frame(container)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Next image grid:").pack(side="left", padx=(0, 8))
        self.grid_control = ttk.Combobox(toolbar, textvariable=self._grid_var,
                                        values=list(self.GRID_LABELS), state="readonly", width=17,
                                        font=(self._font_family, 10))
        self.grid_control.pack(side="left", padx=(0, 10))
        ttk.Button(toolbar, text="Load image", command=self.load_image,
                   style="Accent.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Try demo", command=self.load_demo).pack(side="left", padx=(0, 6))
        self.hint_button = ttk.Button(toolbar, text="Hint (3 left)", command=self.show_hint,
                                      state="disabled")
        self.hint_button.pack(side="left", padx=(0, 6))
        self.solve_button = ttk.Button(toolbar, text="Solve", command=self.solve,
                                       state="disabled")
        self.solve_button.pack(side="left")

        mode_row = ttk.Frame(container)
        mode_row.pack(fill="x", pady=(10, 8))
        ttk.Checkbutton(mode_row, text="Timed challenge for next image (5 / 8 / 10 minutes)",
                        variable=self._challenge_var).pack(side="left")
        ttk.Label(mode_row, textvariable=self._file_var).pack(side="right")

        stats = ttk.Frame(container)
        stats.pack(fill="x", pady=(0, 12))
        for label, variable in [("MOVES", self._moves_var), ("TILES LEFT", self._tiles_var),
                                ("HINTS LEFT", self._hints_var), ("TIME", self._time_var)]:
            card = tk.Frame(stats, bg="white", padx=18, pady=9)
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            tk.Label(card, text=label, bg="white", fg="#65788a",
                     font=(self._font_family, 9)).pack(anchor="w")
            tk.Label(card, textvariable=variable, bg="white", fg="#17354b",
                     font=(self._font_family, 17, "bold")).pack(anchor="w")

        boards = ttk.Frame(container)
        boards.pack()
        canvases = []
        for column, title in enumerate(("ORIGINAL  /  Reference", "YOUR PUZZLE  /  Play here")):
            panel = ttk.Frame(boards)
            panel.grid(row=0, column=column, padx=(0, 16) if column == 0 else (0, 0))
            ttk.Label(panel, text=title, font=(self._font_family, 10, "bold")).pack(anchor="w", pady=(0, 8))
            canvas = tk.Canvas(panel, width=self._canvas_size, height=self._canvas_size,
                               bg="white", highlightthickness=1, highlightbackground="#cfdae3")
            canvas.pack()
            canvas.create_text(self._canvas_size // 2, self._canvas_size // 2,
                               text="Load an image to begin", fill="#708496",
                               font=(self._font_family, 12))
            canvases.append(canvas)
        self.original_canvas, self.puzzle_canvas = canvases
        self.puzzle_canvas.bind("<Button-1>", self._on_left_click)
        self.puzzle_canvas.bind("<Shift-Button-1>", self._on_flip)
        self.puzzle_canvas.bind("<Button-3>", self._on_rotate)
        # Control-click is also available for a Mac with a single-button mouse.
        self.puzzle_canvas.bind("<Control-Button-1>", self._on_rotate)

        self._instructions = ttk.Label(container,
                  text="Left click: select / swap     Right click: turn 90 degrees"
                  "     Shift + left click: flip horizontally", font=(self._font_family, 10),
                  wraplength=2 * self._canvas_size)
        self._instructions.pack(pady=(14, 6))
        self._legend = ttk.Label(container, text="Green tick = correct position and orientation."
                  "     Blue circles = hinted tile and its home.", font=(self._font_family, 9),
                  wraplength=2 * self._canvas_size)
        self._legend.pack()
        self._status_label = ttk.Label(container, textvariable=self._status_var,
                  font=(self._font_family, 10, "bold"), wraplength=2 * self._canvas_size)
        self._status_label.pack(anchor="w", pady=(12, 0))

    def _fit_window_to_screen(self) -> None:
        # Account for the actual font sizes, including Windows display scaling.
        # Text may wrap again after shrinking, so allow a few layout passes.
        for _ in range(4):
            self.root.update_idletasks()
            overflow = self.root.winfo_reqheight() - (self.root.winfo_screenheight() - 90)
            if overflow <= 0 or self._board_limit <= 120:
                break
            self._board_limit = max(120, self._board_limit - overflow - 8)
            self._canvas_size = self._board_limit + 16
            for canvas in (self.original_canvas, self.puzzle_canvas):
                canvas.configure(width=self._canvas_size, height=self._canvas_size)
                canvas.delete("all")
                canvas.create_text(self._canvas_size // 2, self._canvas_size // 2,
                                   text="Load an image to begin", fill="#708496",
                                   font=(self._font_family, 12))
            for label in (self._instructions, self._legend, self._status_label):
                label.configure(wraplength=2 * self._canvas_size)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Choose a puzzle image",
                    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if path:  # Cancelling leaves the current game untouched.
            self.start_round(path)

    def load_demo(self) -> None:
        self.start_round(Path(__file__).resolve().parent / "assets" / "demo.png")

    def start_round(self, path: str | Path) -> bool:
        # Build a candidate before replacing the old round, so bad loads are safe.
        try:
            grid_size = self.GRID_LABELS[self._grid_var.get()]
            original = ImageLoader.load(path, grid_size, self._board_limit)
            candidate = Puzzle(original, grid_size)
        except (OSError, ValueError, KeyError, cv2.error) as error:
            messagebox.showerror("Cannot load image", str(error), parent=self.root)
            return False
        self._puzzle = candidate
        self._started_at = self._clock()
        self._finished_at = None
        self._expired = False
        self._completion_shown = False
        self._time_limit = self.TIME_LIMITS[grid_size] if self._challenge_var.get() else None
        self._file_var.set(f"{Path(path).name}  |  {grid_size} x {grid_size}")
        self._status_var.set("Restore the picture on the right. Grid and timer changes apply to the next image.")
        self._render()
        self._update_stats()
        self._update_time()
        return True

    def _can_play(self) -> bool:
        # Check the deadline during input as well as on each timer callback.
        self._update_time()
        return self._puzzle is not None and not self._puzzle.solved and not self._expired

    def _index_at(self, x: int, y: int) -> int | None:
        if self._puzzle is None:
            return None
        local_x, local_y = x - self._board_x, y - self._board_y
        if not (0 <= local_x < self._puzzle.side and 0 <= local_y < self._puzzle.side):
            return None
        column = int(local_x // self._puzzle.tile_size)
        row = int(local_y // self._puzzle.tile_size)
        return row * self._puzzle.grid_size + column

    def _on_left_click(self, event: tk.Event) -> str:
        if self._can_play():
            index = self._index_at(event.x, event.y)
            if index is not None:
                moved = self._puzzle.select(index)
                self._after_interaction(moved)
        return "break"

    def _on_rotate(self, event: tk.Event) -> str:
        return self._tile_action(event, "rotate")

    def _on_flip(self, event: tk.Event) -> str:
        return self._tile_action(event, "flip")

    def _tile_action(self, event: tk.Event, action_name: str) -> str:
        if self._can_play():
            index = self._index_at(event.x, event.y)
            if index is not None:
                action = RotateTransformation(index) if action_name == "rotate" else FlipTransformation(index)
                self._after_interaction(self._puzzle.move(action))
        return "break"

    def _after_interaction(self, moved: bool) -> None:
        self._render()
        self._update_stats()
        if moved:
            self._status_var.set("Move recorded. Select two tiles to swap, or turn / flip one tile.")
            if self._puzzle.solved and not self._completion_shown:
                self._finished_at = self._clock()
                self._completion_shown = True
                self._update_time()
                self._status_var.set("Picture restored! Load another image to play again.")
                messagebox.showinfo("Puzzle complete", f"Well done! You restored the picture in"
                                    f" {self._puzzle.moves} moves.", parent=self.root)
        elif self._puzzle.selected is not None:
            self._status_var.set("Tile selected. Click another tile to swap, or this tile again to deselect.")
        else:
            self._status_var.set("Selection cleared. Choose a tile to continue.")

    def show_hint(self) -> None:
        if not self._can_play():
            return
        hint = self._puzzle.request_hint()
        if hint is not None:
            self._status_var.set("Blue circles show one incorrect tile and its correct home. They clear after your next move.")
            self._render()
            self._update_stats()

    def solve(self) -> None:
        if self._puzzle is None:
            return
        self._puzzle.solve()
        self._finished_at = self._started_at  # Reset the optional time display too.
        self._completion_shown = True
        self._status_var.set("Puzzle solved automatically. Moves and tiles left reset to zero. Load another image to continue.")
        self._render()
        self._update_stats()
        self._update_time()
        messagebox.showinfo("Puzzle solved", "The remaining transformations were undone."
                            " Moves and tiles left are now zero.", parent=self.root)

    def _update_stats(self) -> None:
        puzzle = self._puzzle
        if puzzle is None:
            return
        self._moves_var.set(str(puzzle.moves))
        self._tiles_var.set(str(puzzle.tiles_left))
        self._hints_var.set(str(puzzle.hints_remaining))
        self.hint_button.configure(text=f"Hint ({puzzle.hints_remaining} left)")
        can_hint = not puzzle.solved and not self._expired and puzzle.hints_remaining > 0
        self.hint_button.configure(state="normal" if can_hint else "disabled")
        self.solve_button.configure(state="disabled" if puzzle.solved else "normal")

    def _render(self) -> None:
        if self._puzzle is None:
            return
        self._photos.clear()
        for canvas, pixels in ((self.original_canvas, self._puzzle.original),
                               (self.puzzle_canvas, self._puzzle.assemble())):
            canvas.delete("all")
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)),
                                       master=self.root)
            self._photos.append(photo)  # Tk needs these references to remain alive.
            canvas.create_image(self._board_x, self._board_y, anchor="nw", image=photo, tags="board")
        side, size = self._puzzle.side, self._puzzle.tile_size
        for line in range(1, self._puzzle.grid_size):
            offset = line * size
            self.puzzle_canvas.create_line(self._board_x + offset, self._board_y,
                    self._board_x + offset, self._board_y + side, fill="#b9c4ce", tags="grid")
            self.puzzle_canvas.create_line(self._board_x, self._board_y + offset,
                    self._board_x + side, self._board_y + offset, fill="#b9c4ce", tags="grid")
        for index in range(self._puzzle.grid_size ** 2):
            if index not in self._puzzle.incorrect_indices:
                row, column = divmod(index, self._puzzle.grid_size)
                x = self._board_x + column * size + size - 19
                y = self._board_y + row * size + 6
                self.puzzle_canvas.create_oval(x - 2, y - 2, x + 15, y + 15,
                                              fill="white", outline="", tags="tick")
                self.puzzle_canvas.create_line(x + 1, y + 7, x + 5, y + 11, x + 12, y + 2,
                                              fill="#188c52", width=3, tags="tick")
        if self._puzzle.selected is not None:
            row, column = divmod(self._puzzle.selected, self._puzzle.grid_size)
            x, y = self._board_x + column * size, self._board_y + row * size
            self.puzzle_canvas.create_rectangle(x + 2, y + 2, x + size - 2, y + size - 2,
                                                outline="#e39422", width=4, tags="selection")
        if self._puzzle.hint is not None:
            position, home = self._puzzle.hint
            self._draw_hint_circle(self.puzzle_canvas, position)
            self._draw_hint_circle(self.original_canvas, home)

    def _draw_hint_circle(self, canvas: tk.Canvas, index: int) -> None:
        size = self._puzzle.tile_size
        row, column = divmod(index, self._puzzle.grid_size)
        x = self._board_x + (column + 0.5) * size
        y = self._board_y + (row + 0.5) * size
        radius = size * 0.30
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                           outline="white", width=6, tags="hint")
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                           outline="#1677ee", width=3, tags="hint")

    def _schedule_clock(self) -> None:
        self._timer_id = self.root.after(250, self._clock_tick)

    def _clock_tick(self) -> None:
        self._update_time()
        self._schedule_clock()

    def _update_time(self) -> None:
        if self._puzzle is None:
            return
        now = self._finished_at if self._finished_at is not None else self._clock()
        elapsed = max(0, int(now - self._started_at))
        if self._time_limit is None:
            seconds = elapsed
        else:
            seconds = max(0, self._time_limit - elapsed)
        self._time_var.set(f"{seconds // 60:02d}:{seconds % 60:02d}")
        if (self._time_limit is not None and elapsed >= self._time_limit
                and not self._expired and not self._puzzle.solved):
            self._expired = True
            self._finished_at = self._started_at + self._time_limit
            self._status_var.set("Time is up. Use Solve to view the answer, or load a new image.")
            self._update_stats()
            messagebox.showinfo("Time is up", "The timed challenge has ended."
                                " Use Solve or load another image.", parent=self.root)

    def close(self) -> None:
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PuzzleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
