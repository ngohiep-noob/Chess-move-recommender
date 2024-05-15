import torch
import chess
from torch.utils.data import Dataset
from typing import List, Tuple

from utils import *


class ChessDataset(Dataset):
    def __init__(
        self, moves: List[Tuple[str, str, str, chess.Board]], mask_move: bool = False
    ):
        self.moves = moves
        self.mask_move = mask_move

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        _from, _to, piece, board_fen = self.moves[idx]
        board = chess.Board(board_fen)
        x = (
            transform_board(board)
            if not self.mask_move
            else transform_board(board, mask_loc=_to)
        )
        # label is one-hot encoded _from
        y = torch.zeros(64)
        y[chess.parse_square(_from)] = 1
        # if play side is black, flip the board both two ways and the label
        if board.turn == chess.BLACK:
            x = x.flip(1).flip(2)
            y = y.flip(0)

        return x, y


class ChessGroupedDataset(Dataset):
    def __init__(self, grouped_moves: List[Tuple[str, dict]]):
        self.grouped_moves = grouped_moves

    def __len__(self):
        return len(self.grouped_moves)

    def __getitem__(self, idx):
        board_fen, piece_prob = self.grouped_moves[idx]
        board = chess.Board(board_fen)
        # feature is the board tensor
        x = transform_board(board)
        # label is one-hot encoded _from
        y = piece_distribution_to_label(piece_prob)
        # if play side is black, flip the board both two ways and the label
        if board.turn == chess.BLACK:
            x = x.flip(1).flip(2)
            y = y.flip(0)

        return x, y
