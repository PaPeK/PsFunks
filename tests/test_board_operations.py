from ps_funks import board_operations as bo
from pathlib import Path

_d_test_data = Path(__file__).resolve().parents[0] / 'data'

def test_BoardData():
    board_1 = _d_test_data / "test_board_1.csv"
    board_2 = _d_test_data / "test_board_2.csv"
    board_data = bo.BoardData(test_size=0.33, random_state=42)
    board_data.add_board(board_1, 'board_1')
    board_data.add_board(board_2, 'board_2')
    x, y, dat_name = board_data.get_train_xy('board_1')
    x, y, dat_name = board_data.get_test_xy('board_1')
    x, y, dat_name = board_data.get_strat_xy('board_1')
