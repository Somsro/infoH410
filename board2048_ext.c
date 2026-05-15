#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BOARD_SIZE 16

/**
 * BoardStruct
 * 
 * Represents the internal data structure for the game board.
 */
typedef struct {
    uint16_t data[BOARD_SIZE];
} BoardStruct;

/**
 * BoardObject
 * 
 * Represents the Python object for the game board.
 */
typedef struct {
    PyObject_HEAD
    BoardStruct board;
} BoardObject;

static PyTypeObject BoardType;

/**
 * __new__(cls)
 * 
 * Allocates memory for a new BoardObject and initializes the internal board data to zero.
 */
static PyObject *
Board_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BoardObject *self = (BoardObject *)type->tp_alloc(type, 0);
    if (self != NULL)
        memset(self->board.data, 0, sizeof(self->board.data));
    return (PyObject *)self;
}

/**
 * __init__(self)
 * 
 * Initializes the board to the starting state of the game, with two tiles
 * containing the value 2 (represented as 1 in the internal data).
 * 
 * The initial positions of the two tiles are randomly chosen among the 16 available cells.
 */
static int
Board_init(BoardObject *self, PyObject *args, PyObject *kwds)
{
    // Place two tiles in random positions on the board
    for (int i = 0; i < 2; i++) {
        int pos;
        do {
            pos = rand() % BOARD_SIZE;
        } while (self->board.data[pos] != 0); // Ensure we place on an empty cell
        self->board.data[pos] = 1; // Place a tile
    }
    return 0;
}

/**
 * Board_sweep_struct(board, direction)
 *
 * Applies the sweep logic to a raw BoardStruct.
 * Returns 1 if the board changed, 0 if it stayed the same, and -1 on invalid direction.
 */
static int
Board_sweep_struct(BoardStruct *board, int direction)
{
    if (direction < 0 || direction > 3) {
        PyErr_SetString(PyExc_ValueError, "Direction must be 0 (up), 1 (right), 2 (down), or 3 (left)");
        return -1;
    }

    int lane_start, lane_step, cell_step;

    switch (direction) {
        case 0: lane_start = 0;  lane_step = 1; cell_step =  4; break; // up
        case 1: lane_start = 3;  lane_step = 4; cell_step = -1; break; // right
        case 2: lane_start = 12; lane_step = 1; cell_step = -4; break; // down
        case 3: lane_start = 0;  lane_step = 4; cell_step =  1; break; // left
    }

    int changed = 0;

    for (int i = 0; i < 4; i++) {
        int base = lane_start + i * lane_step;

        for (int j = 0; j < 3; j++) {
            int pos = base + j * cell_step;

            for (int k = 0; k < 3 - j; k++) {
                int check = base + (j + k + 1) * cell_step;
                if (board->data[check] == 0)
                    continue;

                if (board->data[pos] == 0) {
                    board->data[pos] = board->data[check];
                    board->data[check] = 0;
                    changed = 1;
                } else if (board->data[pos] == board->data[check]) {
                    board->data[pos]++;
                    board->data[check] = 0;
                    changed = 1;
                    break;
                } else {
                    break;
                }
            }
        }
    }

    return changed;
}

/**
 * Board_sweep(self, direction)
 * 
 * Sweeps the board in the given direction (0=up, 1=right, 2=down, 3=left).
 * Returns True if any tile moved or merged, False if the board was unchanged.
 */
static PyObject *
Board_sweep(BoardObject *self, PyObject *args)
{
    int direction;
    if (!PyArg_ParseTuple(args, "i", &direction))
        return NULL;

    int changed = Board_sweep_struct(&self->board, direction);
    if (changed < 0)
        return NULL;

    return PyBool_FromLong(changed);
}

/**
 * Board_is_move_valid(self, direction)
 *
 * Returns True if calling sweep(direction) would change the board.
 */
static PyObject *
Board_is_move_valid(BoardObject *self, PyObject *args)
{
    int direction;
    if (!PyArg_ParseTuple(args, "i", &direction))
        return NULL;

    BoardStruct copy;
    memcpy(&copy, &self->board, sizeof(BoardStruct));

    int changed = Board_sweep_struct(&copy, direction);
    if (changed < 0)
        return NULL;

    return PyBool_FromLong(changed);
}

/**
 * Board_place_tile(self)
 * 
 * Places a new tile on a random empty cell. The value is 1 (90% chance) or 2 (10% chance).
 * When the board is full, it checks if any moves are still possible.
 * Returns False if the board is full and there are no possible moves.
 */
static PyObject *
Board_place_tile(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    // Collect all empty cell indices
    int empty[BOARD_SIZE];
    int empty_count = 0;

    for (int i = 0; i < BOARD_SIZE; i++) {
        if (self->board.data[i] == 0)
            empty[empty_count++] = i;
    }

    // Pick a random empty cell and place tile
    int pos = empty[rand() % empty_count];
    self->board.data[pos] = (rand() % 10 == 0) ? 2 : 1;

    // Check if board is full after placement
    int full = 1;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (self->board.data[i] == 0) {
            full = 0;
            break;
        }
    }

    if (!full)
        return PyBool_FromLong(1);  // still empty cells, game continues

    // Board is full — check if any move is still possible
    // by testing all 4 directions on a copy of the board
    for (int direction = 0; direction < 4; direction++) {
        // Copy the board
        BoardStruct copy;
        memcpy(&copy, &self->board, sizeof(BoardStruct));

        // Determine sweep parameters (same logic as Board_sweep)
        int lane_start, lane_step, cell_step;
        switch (direction) {
            case 0: lane_start = 0;  lane_step = 1; cell_step =  4; break; // up
            case 1: lane_start = 3;  lane_step = 4; cell_step = -1; break; // right
            case 2: lane_start = 12; lane_step = 1; cell_step = -4; break; // down
            case 3: lane_start = 0;  lane_step = 4; cell_step =  1; break; // left
        }

        int changed = 0;
        for (int i = 0; i < 4 && !changed; i++) {
            int base = lane_start + i * lane_step;
            for (int j = 0; j < 3 && !changed; j++) {
                int pos = base + j * cell_step;
                for (int k = 0; k < 3 - j; k++) {
                    int check = base + (j + k + 1) * cell_step;
                    if (copy.data[check] == 0)
                        continue;
                    if (copy.data[pos] == 0 || copy.data[pos] == copy.data[check]) {
                        changed = 1;  // a move or merge is possible
                        break;
                    } else {
                        break;
                    }
                }
            }
        }

        if (changed)
            return PyBool_FromLong(1);  // at least one valid move exists
    }

    return PyBool_FromLong(0);  // no moves possible, game over
}

/**
 * Board_force_place_tile(self, pos, log_value)
 *
 * Forces a tile to be placed at a specific position with a specific log value.
 */
static PyObject *
Board_force_place_tile(BoardObject *self, PyObject *args)
{
    int pos, log_value;
    if (!PyArg_ParseTuple(args, "ii", &pos, &log_value))
        return NULL;

    if (pos < 0 || pos >= BOARD_SIZE) {
        PyErr_SetString(PyExc_ValueError, "Position must be between 0 and 15");
        return NULL;
    }
    if (log_value < 0) {
        PyErr_SetString(PyExc_ValueError, "Log value must be non-negative");
        return NULL;
    }
    if (self->board.data[pos] != 0) {
        PyErr_SetString(PyExc_ValueError, "Cell is not empty");
        return NULL;
    }

    self->board.data[pos] = (uint16_t)log_value;
    return PyBool_FromLong(1);
}

/**
 * Board_get_empty_cells(self)
 *
 * Returns a list of indices of all empty cells on the board.
 */
static PyObject *
Board_get_empty_cells(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *list = PyList_New(0);
    if (!list) return NULL;

    for (int i = 0; i < BOARD_SIZE; i++) {
        if (self->board.data[i] == 0) {
            PyObject *index = PyLong_FromLong(i);
            if (!index) {
                Py_DECREF(list);
                return NULL;
            }
            if (PyList_Append(list, index) < 0) {
                Py_DECREF(index);
                Py_DECREF(list);
                return NULL;
            }
            Py_DECREF(index);
        }
    }

    return list;
}

/**
 * Board_repr(self)
 * 
 * Returns a string representation of the board.
 */
static PyObject *
Board_repr(BoardObject *self)
{
    /* Build "Board([v0, v1, ..., v15])" */
    PyObject *list = PyList_New(BOARD_SIZE);
    if (!list) return NULL;
    for (int i = 0; i < BOARD_SIZE; i++)
        PyList_SET_ITEM(list, i, PyLong_FromLong(self->board.data[i]));
    PyObject *list_repr = PyObject_Repr(list);
    Py_DECREF(list);
    if (!list_repr) return NULL;
    PyObject *result = PyUnicode_FromFormat("Board(%S)", list_repr);
    Py_DECREF(list_repr);
    return result;
}

/**
 * Board_to_list(self)
 * 
 * Returns a list representation of the board.
 */
static PyObject *
Board_to_list(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *list = PyList_New(BOARD_SIZE);
    if (!list) return NULL;
    for (int i = 0; i < BOARD_SIZE; i++)
        PyList_SET_ITEM(list, i, PyLong_FromLong(self->board.data[i]));
    return list;
}

/**
 * Board_max_log_tile(self)
 *
 * Internal helper: returns the maximum log2 tile value on the board.
 */
static uint16_t
Board_max_log_tile(const BoardObject *self)
{
    uint16_t max_log_tile = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (self->board.data[i] > max_log_tile)
            max_log_tile = self->board.data[i];
    }
    return max_log_tile;
}

/**
 * Board_get_emptyCount(self)
 *
 * Returns the number of empty tiles on the board.
 */
static PyObject *
Board_get_emptyCount(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    int empty_count = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (self->board.data[i] == 0)
            empty_count++;
    }
    return PyLong_FromLong(empty_count);
}

/**
 * Board_get_max_logTile(self)
 *
 * Returns log2 of the maximum tile value present on the board.
 * The internal representation already stores log2(tile), so we return max cell value.
 */
static PyObject *
Board_get_max_logTile(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    uint16_t max_log_tile = Board_max_log_tile(self);
    return PyLong_FromUnsignedLong((unsigned long)max_log_tile);
}

/**
 * Board_is_max_corner(self)
 *
 * Returns True if at least one maximum tile is located in a corner.
 */
static PyObject *
Board_is_max_corner(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    uint16_t max_log_tile = Board_max_log_tile(self);

    int in_corner =
        (self->board.data[0] == max_log_tile) ||
        (self->board.data[3] == max_log_tile) ||
        (self->board.data[12] == max_log_tile) ||
        (self->board.data[15] == max_log_tile);

    return PyBool_FromLong(in_corner);
}

/**
 * Board_clone(self)
 *
 * Returns a copy of the board.
 */
static PyObject *
Board_clone(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    BoardObject *clone = (BoardObject *)BoardType.tp_new(&BoardType, NULL, NULL);
    if (!clone) return NULL;
    memcpy(clone->board.data, self->board.data, sizeof(self->board.data));
    return (PyObject *)clone;
}

/**
 * Board_get_monotonicity(self)
 *
 * Returns the monotonicity score of the board. Calculated as the sum of monotonicity in rows and columns.
 * Higher (more positive) is more monotone.
 */
static PyObject *
Board_get_monotonicity(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    int mono_left = 0, mono_right = 0, mono_up = 0, mono_down = 0;

    // Check rows for left/right monotonicity
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (self->board.data[i * 4 + j] > self->board.data[i * 4 + j + 1])
                mono_left++;
            else if (self->board.data[i * 4 + j] < self->board.data[i * 4 + j + 1])
                mono_right++;
        }
    }

    // Check columns for up/down monotonicity
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 3; i++) {
            if (self->board.data[i * 4 + j] > self->board.data[(i + 1) * 4 + j])
                mono_up++;
            else if (self->board.data[i * 4 + j] < self->board.data[(i + 1) * 4 + j])
                mono_down++;
        }
    }

    // Return the monoticity as a score: higher is more monotone
    int mono_score = max(mono_left, mono_right) + max(mono_up, mono_down);
    return PyLong_FromLong(mono_score);
}

/**
 * Board_get_smoothness(self)
 * 
 * Returns the smoothness score of the board, calculated as the negative sum of absolute differences between adjacent tiles. 
 * Higher (less negative) is smoother.
 */
static PyObject *
Board_get_smoothness(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    int smoothness = 0;

    // Check rows for smoothness
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            int a = self->board.data[i * 4 + j];
            int b = self->board.data[i * 4 + j + 1];
            if (a > 0 && b > 0) {
                smoothness -= abs(a - b);  // compare log2 values directly
            }
        }
    }

    // Check columns for smoothness
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 3; i++) {
            int a = self->board.data[i * 4 + j];
            int b = self->board.data[(i + 1) * 4 + j];
            if (a > 0 && b > 0) {
                smoothness -= abs(a - b);  // compare log2 values directly
            }
        }
    }

    return PyLong_FromLong(smoothness);
}

/**
 * Board_get_merge_potential(self)
 *
 * Returns the number of potential merges on the board.
 */
static PyObject *
Board_get_merge_potential(BoardObject *self, PyObject *Py_UNUSED(ignored))
{
    int merges = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            uint16_t a = self->board.data[i * 4 + j];
            uint16_t b = self->board.data[i * 4 + j + 1];
            if (a != 0 && a == b) merges++;

            a = self->board.data[j * 4 + i];
            b = self->board.data[(j + 1) * 4 + i];
            if (a != 0 && a == b) merges++;
        }
    }
    return PyLong_FromLong(merges);
}

/**
 * Board_methods
 * 
 * Defines the methods available on the Board object.
 */
static PyMethodDef Board_methods[] = {
    {
        "sweep",
        (PyCFunction)Board_sweep,
        METH_VARARGS,
        "sweep(direction)\n\n"
        "Sweep the board in the given direction (0=up, 1=right, 2=down, 3=left).\n"
        "Returns True if any tile moved or merged, False if the board was unchanged."
    },
    {
        "place_tile",
        (PyCFunction)Board_place_tile,
        METH_NOARGS,
        "place_tile()\n\n"
        "Place a new tile on a random empty cell.\n"
        "Value is 1 (90% chance) or 2 (10% chance).\n"
        "Returns True if a tile was placed, False if the board is full."
    },
    {
        "force_place_tile",
        (PyCFunction)Board_force_place_tile,
        METH_VARARGS,
        "force_place_tile(pos, log_value)\n\n"
        "Place a tile with log2 value at the specified position (0-15).\n"
        "Returns True if the tile was placed, False if the cell is not empty."
    },
    {
        "get_empty_cells",
        (PyCFunction)Board_get_empty_cells,
        METH_NOARGS,
        "get_empty_cells()\n\n"
        "Return a list of indices of empty cells on the board."
    },
    {
        "is_move_valid",
        (PyCFunction)Board_is_move_valid,
        METH_VARARGS,
        "is_move_valid(direction)\n\n"
        "Return True if sweep(direction) would change the board."
    },
    {
        "to_list",
        (PyCFunction)Board_to_list,
        METH_NOARGS,
        "to_list()\n\n"
        "Return the contents as a Python list of ints."
    },
    {
        "get_emptyCount",
        (PyCFunction)Board_get_emptyCount,
        METH_NOARGS,
        "get_emptyCount()\n\n"
        "Return the number of empty tiles on the board."
    },
    {
        "get_max_logTile",
        (PyCFunction)Board_get_max_logTile,
        METH_NOARGS,
        "get_max_logTile()\n\n"
        "Return log2 of the largest tile on the board."
    },
    {
        "is_max_corner",
        (PyCFunction)Board_is_max_corner,
        METH_NOARGS,
        "is_max_corner()\n\n"
        "Return True if the largest tile is in a corner."
    },
    {
        "clone",
        (PyCFunction)Board_clone,
        METH_NOARGS,
        "clone()\n\n"
        "Return a new Board object with the same state."
    },
    {
        "get_monotonicity",
        (PyCFunction)Board_get_monotonicity,
        METH_NOARGS,
        "get_monotonicity()\n\n"
        "Return the monotonicity score of the board."
    },
    {
        "get_smoothness",
        (PyCFunction)Board_get_smoothness,
        METH_NOARGS,
        "get_smoothness()\n\n"
        "Return the smoothness score of the board."
    },
    {
        "get_merge_potential",
        (PyCFunction)Board_get_merge_potential,
        METH_NOARGS,
        "get_merge_potential()\n\n"
        "Return the number of potential merges on the board."
    },
    {NULL, NULL, 0, NULL}   /* sentinel */
};

/**
 * BoardType
 * 
 * The Python type object for the Board class.
 */
static PyTypeObject BoardType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "board2048_ext.Board",
    .tp_doc       = PyDoc_STR(
        "Board() -> new Board\n\n"
        "A 4x4 game board for the 2048 game.\n\n"
    ),
    .tp_basicsize = sizeof(BoardObject),
    .tp_itemsize  = 0,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = Board_new,
    .tp_init      = (initproc)Board_init,
    .tp_repr      = (reprfunc)Board_repr,
    .tp_methods   = Board_methods,
};

/**
 * board2048_module
 */
static PyModuleDef board2048_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "board2048_ext",
    .m_doc = "C extension module for a 2048 game board.",
    .m_size = -1,
};

/**
 * PyInit_board2048_ext
 */
PyMODINIT_FUNC
PyInit_board2048_ext(void)
{
    srand((unsigned int)time(NULL)); // Seed the random number generator

    PyObject *m;
    if (PyType_Ready(&BoardType) < 0)
        return NULL;

    m = PyModule_Create(&board2048_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BoardType);
    if (PyModule_AddObject(m, "Board", (PyObject *)&BoardType) < 0) {
        Py_DECREF(&BoardType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}