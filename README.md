# Comparisons of different AI techniques on the game 2048

## Prerequisities
You need to compile the C extension for the board using the following command:
> make

## Training
To train TD and DQN agents and run each agent on one run (after training or load the saved one directly) run those commands:
> make expectimax\
> make dqn\
> make td

The training datas tracking can be found in the tracking folder.

## Test
The following commands will test the different approaches using saved models on 100runs, store the results in a npz file and store the mean/std/min/max values in a txt file in the tarcking folder:
> make testExpectimax\
> make testDQN\
> make testTD

## Statistical tests
If the tracking files (dfrom previous section) are present in the tracking folder you can run this command to execute pairwise tests on approaches (for comparison).
> make stats