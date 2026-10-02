from __future__ import annotations
from pathlib import Path

if __name__ == "__main__":
    import subprocess
    import sys

    print("Starting the puzzle through app.py. Next time, open start.command.")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("app.py")), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(result.returncode)

import random

import cv2
import numpy as np


_SCRAMBLE_COUNTS = {3: 6, 4: 12, 5: 20}


class ImageProcessor:
    
    SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

    @staticmethod
    def load_and_prepare(path: str | Path, grid_size: int, max_side: int = 600) -> np.ndarray:
        
        if grid_size not in _SCRAMBLE_COUNTS:
            raise ValueError("Grid size must be 3, 4, or 5.")
        if not isinstance(max_side, int) or max_side < grid_size:
            raise ValueError("Maximum side must be at least the grid size.")

        source_path = Path(path)
        if source_path.suffix.lower() not in ImageProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError("Choose a JPG, PNG, or BMP image.")
        try:
            encoded = np.fromfile(str(source_path), dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"Cannot open image: {source_path}") from exc
        if not encoded.size:
            raise ValueError("The selected image is empty.")

        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("The selected file is not a readable image.")
        if decoded.dtype == np.uint16:
            decoded = (decoded / 257).astype(np.uint8)
        elif decoded.dtype != np.uint8:
            raise ValueError("This image's pixel format is not supported.")

        if decoded.ndim == 2:
            image = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
        elif decoded.shape[2] == 4:
            alpha = decoded[:, :, 3:4].astype(np.float32) / 255.0
            image = np.rint(decoded[:, :, :3] * alpha + 242 * (1 - alpha)).astype(np.uint8)
        elif decoded.shape[2] == 3:
            image = decoded
        else:
            raise ValueError("This image's channels are not supported.")

        height, width = image.shape[:2]
        limit = (max_side // grid_size) * grid_size
        scale = min(1.0, limit / max(height, width))
        new_width = max(1, min(limit, round(width * scale)))
        new_height = max(1, min(limit, round(height * scale)))
        if (new_width, new_height) != (width, height):
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        side = ((max(new_width, new_height) + grid_size - 1) // grid_size) * grid_size
        left = (side - new_width) // 2
        right = side - new_width - left
        top = (side - new_height) // 2
        bottom = side - new_height - top
        return cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(242, 242, 242)
        )


class Tile:

    def __init__(self, home_index: int, pixels: np.ndarray) -> None:
        self._home_index = home_index
        self._pixels = pixels.copy()
        
        self._corners = [0, 1, 2, 3]

    @property
    def home_index(self) -> int:
        return self._home_index

    @property
    def is_upright(self) -> bool:
        return self._corners == [0, 1, 2, 3]

    def rotate_cw(self, quarter_turns: int = 1) -> None:
        for _ in range(quarter_turns % 4):
            self._pixels = cv2.rotate(self._pixels, cv2.ROTATE_90_CLOCKWISE)
            old = self._corners
            self._corners = [old[3], old[0], old[1], old[2]]

    def flip(self, axis: str) -> None:
        old = self._corners
        if axis == "horizontal":
            self._pixels = cv2.flip(self._pixels, 1)
            self._corners = [old[1], old[0], old[3], old[2]]
        elif axis == "vertical":
            self._pixels = cv2.flip(self._pixels, 0)
            self._corners = [old[3], old[2], old[1], old[0]]
        else:
            raise ValueError("Flip axis must be horizontal or vertical.")

    def render(self) -> np.ndarray:
        return self._pixels.copy()

    def matches_pixels(self, pixels: np.ndarray) -> bool:
        return np.array_equal(self._pixels, pixels)


class PuzzleAction:
    PuzzleBoard calls these methods on any action without needing to know its
    specific type. This is inheritance and polymorphism.
    """

    def apply(self, board: PuzzleBoard) -> None:
        raise NotImplementedError

    def undo(self, board: PuzzleBoard) -> None:
        raise NotImplementedError


class SwapAction(PuzzleAction):
    def __init__(self, first: int, second: int) -> None:
        self.first = first
        self.second = second

    def apply(self, board: PuzzleBoard) -> None:
        board._swap_tiles(self.first, self.second)

    def undo(self, board: PuzzleBoard) -> None:
        board._swap_tiles(self.first, self.second)


class RotateAction(PuzzleAction):
    def __init__(self, index: int, quarter_turns: int) -> None:
        self.index = index
        self.quarter_turns = quarter_turns

    def apply(self, board: PuzzleBoard) -> None:
        board._tiles[self.index].rotate_cw(self.quarter_turns)

    def undo(self, board: PuzzleBoard) -> None:
        board._tiles[self.index].rotate_cw(-self.quarter_turns)


class FlipAction(PuzzleAction):
    def __init__(self, index: int, axis: str) -> None:
        self.index = index
        self.axis = axis

    def apply(self, board: PuzzleBoard) -> None:
        board._tiles[self.index].flip(self.axis)

    def undo(self, board: PuzzleBoard) -> None:
        board._tiles[self.index].flip(self.axis)


class PuzzleBoard:

    def __init__(
        self, prepared_bgr: np.ndarray, grid_size: int, rng: random.Random | None = None
    ) -> None:
        if grid_size not in _SCRAMBLE_COUNTS:
            raise ValueError("Grid size must be 3, 4, or 5.")
        if (
            not isinstance(prepared_bgr, np.ndarray)
            or prepared_bgr.dtype != np.uint8
            or prepared_bgr.ndim != 3
            or prepared_bgr.shape[2] != 3
            or prepared_bgr.shape[0] != prepared_bgr.shape[1]
            or prepared_bgr.shape[0] < grid_size
            or prepared_bgr.shape[0] % grid_size
        ):
            raise ValueError("Prepared image must be a square, grid-divisible BGR uint8 image.")

        self._original = prepared_bgr.copy()
        self._grid_size = grid_size
        self._tile_size = prepared_bgr.shape[0] // grid_size
        self._rng = rng if rng is not None else random.Random()
        self._tiles: list[Tile] = []
        for row in range(grid_size):
            for column in range(grid_size):
                top = row * self._tile_size
                left = column * self._tile_size
                patch = self._original[top : top + self._tile_size, left : left + self._tile_size]
                self._tiles.append(Tile(row * grid_size + column, patch))

    
        first_pixels = self._tiles[0].render()
        if (
            all(tile.matches_pixels(first_pixels) for tile in self._tiles)
            and np.array_equal(
                cv2.rotate(first_pixels, cv2.ROTATE_90_CLOCKWISE), first_pixels
            )
            and np.array_equal(cv2.flip(first_pixels, 1), first_pixels)
        ):
            raise ValueError(
                "This image cannot make a visible puzzle. Choose one with more detail."
            )

        for _ in range(80):
            actions = self._generate_scramble_actions()
            for action in actions:
                action.apply(self)
            if self.incorrect_count > 0:
                self._scramble_actions = actions
                break
            for action in reversed(actions):
                action.undo(self)
        else:
            raise ValueError(
                "This image could not make a visible puzzle. Choose one with more detail."
            )
        self._player_actions: list[PuzzleAction] = []
        self._move_count = 0
        self._scramble_active = True

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def original_image(self) -> np.ndarray:
        return self._original.copy()

    @property
    def scramble_actions(self) -> tuple[PuzzleAction, ...]:
        return self._scramble_actions

    @property
    def move_count(self) -> int:
        return self._move_count

    @property
    def incorrect_count(self) -> int:
        return sum(not self.tile_is_correct(index) for index in range(len(self._tiles)))

    @property
    def is_solved(self) -> bool:
        return self.incorrect_count == 0

    def tile_is_correct(self, index: int) -> bool:
        self._validate_index(index)
        tile = self._tiles[index]
        row, column = divmod(index, self._grid_size)
        top = row * self._tile_size
        left = column * self._tile_size
        original_pixels = self._original[
            top : top + self._tile_size, left : left + self._tile_size
        ]
        return tile.matches_pixels(original_pixels)

    def tile_home(self, index: int) -> int:
        self._validate_index(index)
        return self._tiles[index].home_index

    def render_image(self) -> np.ndarray:
        result = np.empty_like(self._original)
        for index, tile in enumerate(self._tiles):
            row, column = divmod(index, self._grid_size)
            top = row * self._tile_size
            left = column * self._tile_size
            result[top : top + self._tile_size, left : left + self._tile_size] = tile.render()
        return result

    def swap(self, first: int, second: int) -> bool:
        self._validate_index(first)
        self._validate_index(second)
        if first == second or self.is_solved:
            return False
        return self._play(SwapAction(first, second))

    def rotate(self, index: int) -> bool:
        self._validate_index(index)
        if self.is_solved:
            return False
        return self._play(RotateAction(index, 1))

    def flip_horizontal(self, index: int) -> bool:
        self._validate_index(index)
        if self.is_solved:
            return False
        return self._play(FlipAction(index, "horizontal"))

    def get_hint(self) -> tuple[int, int] | None:
        incorrect = [index for index in range(len(self._tiles)) if not self.tile_is_correct(index)]
        if not incorrect:
            return None
        current_index = self._rng.choice(incorrect)
        return current_index, self.tile_home(current_index)

    def solve(self) -> None:
        if not self._scramble_active:
            return
        for action in reversed(self._player_actions):
            action.undo(self)
        for action in reversed(self._scramble_actions):
            action.undo(self)
        self._player_actions.clear()
        self._move_count = 0
        self._scramble_active = False

    def _play(self, action: PuzzleAction) -> bool:
        action.apply(self)
        self._player_actions.append(action)
        self._move_count += 1
        return True

    def _swap_tiles(self, first: int, second: int) -> None:
        self._tiles[first], self._tiles[second] = self._tiles[second], self._tiles[first]

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or not 0 <= index < len(self._tiles):
            raise IndexError("Tile index is outside the puzzle grid.")

    def _generate_scramble_actions(self) -> tuple[PuzzleAction, ...]:
        count = _SCRAMBLE_COUNTS[self._grid_size]
        kinds = ["swap", "rotate", "flip"]
        kinds.extend(self._rng.choice(("swap", "rotate", "flip")) for _ in range(count - 3))
        swap_positions = [index for index, kind in enumerate(kinds) if kind == "swap"]
        if len(swap_positions) % 2 == 0:
            # There are at least two swaps here: one mandatory and one extra.
            kinds[swap_positions[-1]] = self._rng.choice(("rotate", "flip"))
        self._rng.shuffle(kinds)

        actions: list[PuzzleAction] = []
        tile_count = self._grid_size * self._grid_size
        for kind in kinds:
            index = self._rng.randrange(tile_count)
            if kind == "swap":
                other = self._rng.randrange(tile_count - 1)
                if other >= index:
                    other += 1
                actions.append(SwapAction(index, other))
            elif kind == "rotate":
                actions.append(RotateAction(index, self._rng.choice((1, 2, 3))))
            else:
                actions.append(FlipAction(index, self._rng.choice(("horizontal", "vertical"))))
        return tuple(actions)
