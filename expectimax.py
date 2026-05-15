INFINITY = int(1e9)
PROB_CUTOFF = 1e-4

class ExpectimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def expectimax(self, env, depth, is_chance_node, prob=1.0):
        if depth == 0 or env.is_done() or prob < PROB_CUTOFF:
            return env.get_score("expectimax")

        if is_chance_node:
            total = 0
            empty_cells = env.get_empty_cells()
            n = len(empty_cells)
            if n == 0:  # No empty cells, don't divide by zero
                return self.expectimax(env, depth - 1, False)
            for cell in empty_cells:
                for (tile, branch_prob) in [(1, 0.9), (2, 0.1)]:
                    p = branch_prob / n  # Probability of this tile appearing in this cell
                    if prob * p < PROB_CUTOFF:
                        continue  # Skip very unlikely outcomes
                    new_env = env.clone()
                    new_env.place_tile(cell, tile)
                    total += p * self.expectimax(new_env, depth - 1, False, prob * p)
            return total
        
        else:
            best = -INFINITY
            for action in env.get_valid_actions():
                new_env = env.clone()
                new_env.simple_step(action)
                score = self.expectimax(new_env, depth - 1, True, prob)
                best = max(best, score)
            return best

    def select_action(self, env, valid_moves):
        best_action, best_score = None, -INFINITY
        for action in valid_moves:
            new_env = env.clone()
            new_env.simple_step(action)
            score = self.expectimax(new_env, self.depth-1, True)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action