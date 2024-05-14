import chess
import chess.pgn
from chess.pgn import BaseVisitor
import io


class Visitor(BaseVisitor):
    def __init__(self):
        self.comments = []
        self.moves = []

    def visit_comment(self, comment: str) -> None:
        self.comments.append(comment)

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.moves.append(move)

    def result(self):
        return self.moves, self.comments


def read_pgn(pgn):
    pgn = io.StringIO(pgn)
    moves, _ = chess.pgn.read_game(pgn, Visitor=Visitor)
    return moves
