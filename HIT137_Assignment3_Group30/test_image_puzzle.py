"""Behaviour checks. Run: python -m unittest -v test_image_puzzle.py

GUI checks create real Tk widgets and generate mouse events. They are skipped
only when a display is unavailable; model/image checks still run headlessly.
"""

import itertools
import random
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from image_puzzle import (FlipTransformation, ImageLoader, Puzzle, PuzzleApp,
                          RotateTransformation, SwapTransformation, Tile)


def board(grid=3, size=60):
    """Asymmetric, reproducible pixels prevent symmetry from hiding mistakes."""
    rng = np.random.default_rng(135)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


class TileTests(unittest.TestCase):
    def test_four_rotations_restore_pixels_and_orientation(self):
        pixels = board(size=9)
        tile = Tile(0, pixels)
        for turn in range(1, 5):
            tile.rotate()
            np.testing.assert_array_equal(tile.pixels, np.rot90(pixels, -turn))
        self.assertTrue(tile.is_correct(0))
        self.assertFalse(tile.is_correct(1))

    def test_double_horizontal_or_vertical_flip_restores(self):
        for axis, numpy_axis in (("horizontal", 1), ("vertical", 0)):
            with self.subTest(axis=axis):
                pixels = board(size=9)
                tile = Tile(0, pixels)
                tile.flip(axis)
                np.testing.assert_array_equal(tile.pixels, np.flip(pixels, numpy_axis))
                self.assertFalse(tile.is_correct(0))
                tile.flip(axis)
                self.assertTrue(tile.is_correct(0))
                np.testing.assert_array_equal(tile.pixels, pixels)

    def test_all_short_rotation_reflection_sequences(self):
        # Independently reconstruct each final state using NumPy, not OpenCV.
        original = board(size=9)
        for length in range(5):
            for sequence in itertools.product(("r", "h", "v"), repeat=length):
                tile = Tile(0, original)
                expected = original.copy()
                for operation in sequence:
                    if operation == "r":
                        tile.rotate()
                        expected = np.rot90(expected, -1)
                    else:
                        tile.flip("horizontal" if operation == "h" else "vertical")
                        expected = np.flip(expected, 1 if operation == "h" else 0)
                turns, reflected = tile.orientation
                canonical = np.fliplr(original) if reflected else original
                canonical = np.rot90(canonical, -turns)
                np.testing.assert_array_equal(tile.pixels, expected)
                np.testing.assert_array_equal(tile.pixels, canonical)
                self.assertEqual(tile.is_correct(0), np.array_equal(expected, original))

    def test_pixels_are_encapsulated(self):
        pixels = board(size=9)
        original = pixels.copy()
        tile = Tile(0, pixels)
        pixels[:] = 0
        returned = tile.pixels
        returned[:] = 0
        np.testing.assert_array_equal(tile.pixels, original)

    def test_invalid_flip_is_rejected_without_change(self):
        tile = Tile(0, board(size=9))
        pixels = tile.pixels
        with self.assertRaises(ValueError):
            tile.flip("diagonal")
        self.assertEqual(tile.orientation, (0, False))
        np.testing.assert_array_equal(tile.pixels, pixels)


class ImageTests(unittest.TestCase):
    def test_all_formats_and_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in (".jpg", ".png", ".bmp"):
                image = board(size=85)
                success, encoded = cv2.imencode(extension, image)
                self.assertTrue(success)
                path = Path(directory) / ("obrázek" + extension)
                path.write_bytes(encoded.tobytes())
                for grid in (3, 4, 5):
                    with self.subTest(extension=extension, grid=grid):
                        prepared = ImageLoader.load(path, grid, 401)
                        self.assertEqual(prepared.shape[0], prepared.shape[1])
                        self.assertEqual(prepared.shape[0] % grid, 0)
                        self.assertLessEqual(prepared.shape[0], 401)
                        puzzle = Puzzle(prepared, grid, random.Random(1))
                        puzzle.solve()
                        np.testing.assert_array_equal(puzzle.assemble(), prepared)

    def test_portrait_landscape_and_tiny_images_keep_proportions(self):
        for height, width in ((120, 240), (240, 120), (1, 1), (1, 150), (150, 1)):
            for grid in (3, 4, 5):
                with self.subTest(height=height, width=width, grid=grid):
                    image = np.full((height, width, 3), (30, 90, 150), np.uint8)
                    result = ImageLoader.prepare(image, grid, 401)
                    side = result.shape[0]
                    scale = side / max(height, width)
                    expected_width = max(1, round(width * scale))
                    expected_height = max(1, round(height * scale))
                    mask = np.all(result == (30, 90, 150), axis=2)
                    ys, xs = np.where(mask)
                    self.assertEqual(xs.max() - xs.min() + 1, expected_width)
                    self.assertEqual(ys.max() - ys.min() + 1, expected_height)

    def test_non_image_empty_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, data in (("notes.txt", b"not an image"), ("empty.png", b""),
                               ("bad.jpg", b"\xff\xd8badjpeg")):
                path = Path(directory) / name
                path.write_bytes(data)
                with self.subTest(name=name), self.assertRaises((ValueError, cv2.error)):
                    ImageLoader.load(path, 3, 400)
            with self.assertRaises(OSError):
                ImageLoader.load(Path(directory) / "missing.png", 3, 400)

    def test_invalid_grid_or_board_is_rejected(self):
        with self.assertRaises(ValueError):
            ImageLoader.prepare(board(), 2, 400)
        with self.assertRaises(ValueError):
            ImageLoader.prepare(board(), 5, 3)
        with self.assertRaises(ValueError):
            Puzzle(np.zeros((60, 61, 3), dtype=np.uint8), 3)
        with self.assertRaises(ValueError):
            Puzzle(board(size=61), 3)


class PuzzleTests(unittest.TestCase):
    def puzzle(self, grid=3, seed=42):
        return Puzzle(board(grid), grid, random.Random(seed))

    def test_scramble_count_all_types_no_duplicate_targets_and_solvable(self):
        for grid, count in ((3, 6), (4, 12), (5, 20)):
            for seed in range(100):
                with self.subTest(grid=grid, seed=seed):
                    puzzle = self.puzzle(grid, seed)
                    actions = puzzle.scramble_actions
                    self.assertEqual(len(actions), count)
                    self.assertEqual({type(action) for action in actions},
                                     {SwapTransformation, RotateTransformation, FlipTransformation})
                    targets = [i for action in actions for i in action.targets]
                    self.assertEqual(len(targets), len(set(targets)))
                    self.assertEqual(set(targets), set(range(grid ** 2)))
                    self.assertEqual(puzzle.tiles_left, grid ** 2)
                    self.assertEqual(puzzle.moves, 0)
                    self.assertFalse(puzzle.solved)
                    puzzle.solve()
                    self.assertTrue(puzzle.solved)
                    np.testing.assert_array_equal(puzzle.assemble(), puzzle.original)

    def test_different_random_seeds_make_different_puzzles(self):
        for grid in (3, 4, 5):
            self.assertFalse(np.array_equal(self.puzzle(grid, 1).assemble(),
                                           self.puzzle(grid, 2).assemble()))

    def test_select_deselect_and_swap_count_one_move(self):
        puzzle = self.puzzle()
        first_home, second_home = puzzle.home_index(0), puzzle.home_index(1)
        self.assertFalse(puzzle.select(0))
        self.assertEqual(puzzle.selected, 0)
        self.assertFalse(puzzle.select(0))
        self.assertIsNone(puzzle.selected)
        self.assertEqual(puzzle.moves, 0)
        puzzle.select(0)
        self.assertTrue(puzzle.select(1))
        self.assertEqual(puzzle.home_index(0), second_home)
        self.assertEqual(puzzle.home_index(1), first_home)
        self.assertEqual(puzzle.moves, 1)
        self.assertIsNone(puzzle.selected)

    def test_hint_limit_and_lifetime(self):
        puzzle = self.puzzle()
        for remaining in (2, 1, 0):
            position, home = puzzle.request_hint()
            self.assertIn(position, puzzle.incorrect_indices)
            self.assertEqual(home, puzzle.home_index(position))
            self.assertEqual(puzzle.hints_remaining, remaining)
        saved_hint = puzzle.hint
        self.assertIsNone(puzzle.request_hint())
        self.assertEqual(puzzle.hint, saved_hint)
        puzzle.select(0)
        puzzle.select(0)
        self.assertEqual(puzzle.hint, saved_hint)  # Selection is not a move.
        self.assertEqual(puzzle.moves, 0)
        puzzle.move(RotateTransformation(0))
        self.assertIsNone(puzzle.hint)
        self.assertEqual(puzzle.moves, 1)

    def test_solve_after_mixed_player_moves_restores_exact_pixels(self):
        for grid in (3, 4, 5):
            puzzle = self.puzzle(grid)
            rng = random.Random(90)
            for _ in range(40):
                index = rng.randrange(grid ** 2)
                kind = rng.choice(("swap", "rotate", "flip"))
                if kind == "swap":
                    second = (index + rng.randrange(1, grid ** 2)) % (grid ** 2)
                    action = SwapTransformation(index, second)
                elif kind == "rotate":
                    action = RotateTransformation(index, rng.randrange(1, 4))
                else:
                    action = FlipTransformation(index, rng.choice(("horizontal", "vertical")))
                puzzle.move(action)
            puzzle.request_hint()
            puzzle.select(0)
            puzzle.solve()
            self.assertTrue(puzzle.solved)
            self.assertEqual(puzzle.moves, 0)
            self.assertEqual(puzzle.tiles_left, 0)
            self.assertIsNone(puzzle.selected)
            self.assertIsNone(puzzle.hint)
            np.testing.assert_array_equal(puzzle.assemble(), puzzle.original)
            puzzle.solve()  # Also safe a second time.
            np.testing.assert_array_equal(puzzle.assemble(), puzzle.original)

    def test_player_can_solve_with_only_specified_controls(self):
        for grid in (3, 4, 5):
            puzzle = self.puzzle(grid)
            for home in range(grid ** 2):
                position = next(i for i in range(grid ** 2) if puzzle.home_index(i) == home)
                if position != home:
                    puzzle.select(position)
                    puzzle.select(home)
            for index in range(grid ** 2):
                turns, flipped = puzzle.orientation(index)
                if flipped:
                    puzzle.move(FlipTransformation(index))
                turns, flipped = puzzle.orientation(index)
                for _ in range((-turns) % 4):
                    puzzle.move(RotateTransformation(index))
            self.assertTrue(puzzle.solved)
            np.testing.assert_array_equal(puzzle.assemble(), puzzle.original)
            moves = puzzle.moves
            self.assertFalse(puzzle.move(RotateTransformation(0)))
            self.assertFalse(puzzle.select(0))
            self.assertIsNone(puzzle.request_hint())
            self.assertEqual(puzzle.moves, moves)

    def test_wrong_orientation_is_still_incorrect_in_right_home(self):
        puzzle = self.puzzle()
        rotation = next(a for a in puzzle.scramble_actions if isinstance(a, RotateTransformation))
        index = rotation.index
        self.assertEqual(puzzle.home_index(index), index)
        self.assertIn(index, puzzle.incorrect_indices)
        puzzle.move(rotation.inverse())
        self.assertNotIn(index, puzzle.incorrect_indices)

    def test_invalid_indices_do_not_change_board_or_moves(self):
        puzzle = self.puzzle()
        pixels = puzzle.assemble()
        for action in (RotateTransformation(-1), FlipTransformation(9), SwapTransformation(0, 9)):
            with self.assertRaises(IndexError):
                puzzle.move(action)
        self.assertEqual(puzzle.moves, 0)
        np.testing.assert_array_equal(puzzle.assemble(), pixels)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            root = tk.Tk()
            root.destroy()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display unavailable: {error}")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.directory.name) / "test.png"
        cv2.imwrite(str(self.image_path), board(size=120))
        self.now = 100.0
        self.info_patch = patch("image_puzzle.messagebox.showinfo")
        self.error_patch = patch("image_puzzle.messagebox.showerror")
        self.info = self.info_patch.start()
        self.error = self.error_patch.start()
        self.root = tk.Tk()
        self.app = PuzzleApp(self.root, clock=lambda: self.now)
        self.root.update()
        self.assertTrue(self.app.start_round(self.image_path))
        self.root.update()

    def tearDown(self):
        self.app.close()
        self.info_patch.stop()
        self.error_patch.stop()
        self.directory.cleanup()

    def click(self, index, sequence="<Button-1>", original=False):
        row, column = divmod(index, self.app.puzzle.grid_size)
        x = self.app._board_x + column * self.app.puzzle.tile_size + 5
        y = self.app._board_y + row * self.app.puzzle.tile_size + 5
        canvas = self.app.original_canvas if original else self.app.puzzle_canvas
        canvas.event_generate(sequence, x=x, y=y)
        self.root.update()

    def test_real_mouse_bindings_and_original_is_reference_only(self):
        self.click(0, original=True)
        self.assertIsNone(self.app.puzzle.selected)
        self.click(0)
        self.assertEqual(self.app.puzzle.selected, 0)
        self.assertEqual(len(self.app.puzzle_canvas.find_withtag("selection")), 1)
        self.click(0)
        self.assertIsNone(self.app.puzzle.selected)
        self.click(0)
        self.click(1)
        self.assertEqual(self.app.puzzle.moves, 1)
        self.click(0, "<Button-3>")
        self.assertEqual(self.app.puzzle.moves, 2)
        self.click(0, "<Shift-Button-1>")
        self.assertEqual(self.app.puzzle.moves, 3)
        self.assertIsNone(self.app.puzzle.selected)  # Shift didn't also select.
        self.assertEqual(self.app._moves_var.get(), "3")
        self.assertEqual(self.app._tiles_var.get(), str(self.app.puzzle.tiles_left))

    def test_boundary_mapping_and_off_image_events_for_every_grid(self):
        for label, grid in self.app.GRID_LABELS.items():
            self.app._grid_var.set(label)
            self.app.start_round(self.image_path)
            self.root.update()
            puzzle = self.app.puzzle
            ox, oy, side = self.app._board_x, self.app._board_y, puzzle.side
            for row in range(grid):
                for column in range(grid):
                    x, y = ox + column * puzzle.tile_size, oy + row * puzzle.tile_size
                    self.assertEqual(self.app._index_at(x, y), row * grid + column)
                    self.assertEqual(self.app._index_at(x + puzzle.tile_size - 1,
                                                      y + puzzle.tile_size - 1), row * grid + column)
            for x, y in ((ox - 1, oy), (ox, oy - 1), (ox + side, oy), (ox, oy + side)):
                self.assertIsNone(self.app._index_at(x, y))
                for sequence in ("<Button-1>", "<Button-3>", "<Shift-Button-1>"):
                    self.app.puzzle_canvas.event_generate(sequence, x=x, y=y)
                    self.root.update()
            self.assertEqual(puzzle.moves, 0)
            self.assertIsNone(puzzle.selected)

    def test_hints_draw_both_circles_limit_and_clear_on_move(self):
        for _ in range(3):
            self.app.hint_button.invoke()
            self.root.update()
            self.assertEqual(len(self.app.original_canvas.find_withtag("hint")), 2)
            self.assertEqual(len(self.app.puzzle_canvas.find_withtag("hint")), 2)
        self.assertIn("disabled", self.app.hint_button.state())
        self.app.hint_button.invoke()
        self.assertEqual(self.app.puzzle.hints_remaining, 0)
        self.click(0)
        self.assertEqual(len(self.app.puzzle_canvas.find_withtag("hint")), 2)
        self.click(1)
        self.assertEqual(len(self.app.original_canvas.find_withtag("hint")), 0)
        self.assertEqual(len(self.app.puzzle_canvas.find_withtag("hint")), 0)

    def test_cancellation_bad_load_and_grid_change_preserve_current_round(self):
        puzzle = self.app.puzzle
        self.click(0)
        self.app.show_hint()
        with patch("image_puzzle.filedialog.askopenfilename", return_value=""):
            self.app.load_image()
        self.assertIs(self.app.puzzle, puzzle)
        self.app._grid_var.set("5 x 5 - Hard")
        self.assertEqual(self.app.puzzle.grid_size, 3)
        invalid = Path(self.directory.name) / "notes.txt"
        invalid.write_text("not an image")
        self.assertFalse(self.app.start_round(invalid))
        self.assertIs(self.app.puzzle, puzzle)
        self.assertEqual(puzzle.selected, 0)
        self.assertEqual(puzzle.hints_remaining, 2)
        self.error.assert_called_once()

    def test_new_round_resets_everything_and_uses_chosen_grid(self):
        self.click(0, "<Button-3>")
        self.app.show_hint()
        self.click(0)
        self.app._grid_var.set("4 x 4 - Medium")
        self.now = 150.0
        self.app.start_round(self.image_path)
        self.assertEqual(self.app.puzzle.grid_size, 4)
        self.assertEqual(self.app.puzzle.moves, 0)
        self.assertEqual(self.app.puzzle.hints_remaining, 3)
        self.assertIsNone(self.app.puzzle.hint)
        self.assertIsNone(self.app.puzzle.selected)
        self.assertEqual(self.app._time_var.get(), "00:00")
        self.assertNotIn("disabled", self.app.hint_button.state())

    def test_solve_button_resets_counters_draws_ticks_and_locks(self):
        self.click(0, "<Button-3>")
        self.app.show_hint()
        self.app.solve_button.invoke()
        self.root.update()
        self.assertTrue(self.app.puzzle.solved)
        self.assertEqual(self.app._moves_var.get(), "0")
        self.assertEqual(self.app._tiles_var.get(), "0")
        self.assertEqual(len(self.app.puzzle_canvas.find_withtag("tick")), 18)
        self.assertIn("disabled", self.app.hint_button.state())
        self.click(0, "<Button-3>")
        self.click(0, "<Shift-Button-1>")
        self.click(0)
        self.assertEqual(self.app.puzzle.moves, 0)
        self.info.assert_called_once()

    def test_player_completion_notification_and_lock(self):
        puzzle = self.app.puzzle
        for action in reversed(puzzle.scramble_actions):
            inverse = action.inverse()
            if isinstance(inverse, SwapTransformation):
                self.click(inverse.first)
                self.click(inverse.second)
            elif isinstance(inverse, RotateTransformation):
                for _ in range(inverse.quarter_turns):
                    self.click(inverse.index, "<Button-3>")
            elif inverse.axis == "horizontal":
                self.click(inverse.index, "<Shift-Button-1>")
            else:
                # A vertical reflection can be undone with H then R180.
                self.click(inverse.index, "<Shift-Button-1>")
                self.click(inverse.index, "<Button-3>")
                self.click(inverse.index, "<Button-3>")
        self.assertTrue(puzzle.solved)
        self.info.assert_called_once()
        self.assertEqual(len(self.app.puzzle_canvas.find_withtag("tick")), 18)
        moves = puzzle.moves
        self.click(0, "<Button-3>")
        self.assertEqual(puzzle.moves, moves)
        self.assertIsNone(puzzle.selected)

    def test_optional_timer_expiry_lock_solve_and_new_round(self):
        self.app._challenge_var.set(True)
        self.app.start_round(self.image_path)
        self.assertEqual(self.app._time_var.get(), "05:00")
        self.now += 299
        self.app._update_time()
        self.assertEqual(self.app._time_var.get(), "00:01")
        self.assertFalse(self.app._expired)
        self.now += 1
        # Input itself must check the limit, even before the timer next fires.
        self.click(0, "<Button-3>")
        self.assertTrue(self.app._expired)
        self.assertEqual(self.app.puzzle.moves, 0)
        self.assertEqual(self.app._time_var.get(), "00:00")
        self.info.assert_called_once()
        self.app.solve_button.invoke()
        self.assertTrue(self.app.puzzle.solved)
        self.app._challenge_var.set(False)
        self.app.start_round(self.image_path)
        self.assertFalse(self.app._expired)
        self.assertEqual(self.app._time_var.get(), "00:00")
        self.click(0, "<Button-3>")
        self.assertEqual(self.app.puzzle.moves, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
