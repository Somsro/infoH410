import sys
from environment import Environment
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

# ── Utility functions ────────────────────────────────────────────────



# ── Game loop ────────────────────────────────────────────────────────

#The idea is to specify the agent_type that will play 2048 and each agent has its class
def main(agent_type) -> None:


    if agent_type == "qlearner":
        agent = Qlearner(
            action_size=4,
            state_size=4896,
            learning_rate=0.0,  # dynamic learning rate
            gamma=0.98,  # does not matter for stateless one-shot game
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.99
        )
    elif agent_type == "expectimax":
        pass  # TODO: implement
    elif agent_type == "deeplearning":
        pass  # TODO: implement
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    env = Environment()

    for i in range(2): # play 2 games to test
        print(f"Game {i+1} starting...")
        done = False
        env.reset()

        while not done: #done is set to True in env.step() when the game is over

            state = env.get_state(agent_type) # get the current state for the agent

            valid_moves = env.get_valid_actions() # get the valid moves for the current state (will always return at least 1 since done would be True otherwise)

            action = agent.select_action(state, valid_moves) # Agents take actions based on the state and valid moves

            # Step in the environment
            obs, reward, done, score = env.step(action) #reward = score difference after taking the action.

            new_state = env.get_state(agent_type) # get the new state after taking the action
            next_valid_actions = env.get_valid_actions() # get the valid moves for the new state (will always return at least 1 since done would be True otherwise)

            # Agent learns
            agent.update(state, action, new_state, reward, score, done, next_valid_actions)

            # Render the new state of the game
            if (not done) :
                env.render(f"Action: {DIRECTION_NAMES[action]}, Actual Score: {score}")
            else:
                env.render(f"Game Over! Final Score: {score}")

        print(f"Game {i+1} ended with score: {score}")
        agent.print_scores(i) # print the score of the episode and the epsilon value to see the progress of the agent



#Je mets ici l'ancien main avant mes changement de qlearners
# def og_main() -> None:
#     board = Board()
#     render(board, "New game! Make your first move.")


#     while True:
#         direction = get_arrow()
#         if direction is None:
#             continue

#         moved = board.sweep(direction)

#         if moved:
#             placed = board.place_tile()
#             if not placed:
#                 render(board, "")
#                 print("GAME OVER — board is full!")
#                 break
#             render(board, f"Moved: {DIRECTION_NAMES[direction]}")
#         else:
#             render(board, f"No change for {DIRECTION_NAMES[direction]} — try another direction.")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            agent_type = sys.argv[1]
            main(agent_type)
        else:
            print("Please specify an agent type.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nBye!")