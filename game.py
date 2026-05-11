import sys
import board2048_ext
from board2048_ext import Board
from qleaners import Qlearner

# ── Platform-specific single-keypress reading ────────────────────────
if sys.platform == "win32":
    import msvcrt

    def get_arrow() -> int | None:
        """Returns 0=up 1=right 2=down 3=left or None for other keys."""
        ch = msvcrt.getch()
        if ch == b'\xe0':
            ch = msvcrt.getch()
            return {b'H': 0, b'M': 1, b'P': 2, b'K': 3}.get(ch)
        if ch == b'\x03':
            raise KeyboardInterrupt
        return None
else:
    import tty
    import termios

    def get_arrow() -> int | None:
        """Returns 0=up 1=right 2=down 3=left or None for other keys."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x03':
                raise KeyboardInterrupt
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == '[':
                    return {'A': 0, 'C': 1, 'B': 2, 'D': 3}.get(ch3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None


# ── Display ──────────────────────────────────────────────────────────

DIRECTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

def render(board: Board, message: str = "") -> None:
    representation = [2**board.to_list()[i] if board.to_list()[i] > 0 else "   ." for i in range(16)]
    for i in range(4):
        row = representation[i*4:(i+1)*4]
        print(' '.join(f"{v:4}" for v in row))
    if message:
        print(message)
    print("Arrow keys to move  •  Ctrl-C to quit")


# ── Game loop ────────────────────────────────────────────────────────

def main() -> None:
    board = Board()
    render(board, "New game! Make your first move.")

    ql_agent = Qlearner(
        action_size=4,
        state_size=65536,
        learning_rate=0.0,  # dynamic learning rate
        gamma=0.98,  # does not matter for stateless one-shot game
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    done = False
    while True:
        # Agents take actions
        action = ql_agent.select_action(state)

        # Step in the environment
        moved = board.sweep(action)
        if moved:
            placed = board.place_tile()
            if not placed:
                render(board, "")
                print("GAME OVER — board is full!")
                ql_agent.update(previous_state, action, state, rewards, dones)
                break
            render(board, f"Moved: {DIRECTION_NAMES[action]}")
        else:
            render(board, f"No change for {DIRECTION_NAMES[action]} — try another direction.")

        # Update Q-tables
        ql_agent.update(previous_state, action, state, rewards, dones)



#Je mets ici l'ancien main avant mes changement de qlearners
def og_main() -> None:
    board = Board()
    render(board, "New game! Make your first move.")


    while True:
        direction = get_arrow()
        if direction is None:
            continue

        moved = board.sweep(direction)

        if moved:
            placed = board.place_tile()
            if not placed:
                render(board, "")
                print("GAME OVER — board is full!")
                break
            render(board, f"Moved: {DIRECTION_NAMES[direction]}")
        else:
            render(board, f"No change for {DIRECTION_NAMES[direction]} — try another direction.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")