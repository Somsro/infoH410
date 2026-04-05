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

    if (direction < 0 || direction > 3) {
        PyErr_SetString(PyExc_ValueError, "Direction must be 0 (up), 1 (right), 2 (down), or 3 (left)");
        return NULL;
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
                if (self->board.data[check] == 0)
                    continue;

                if (self->board.data[pos] == 0) {
                    self->board.data[pos] = self->board.data[check];
                    self->board.data[check] = 0;
                    changed = 1;
                } else if (self->board.data[pos] == self->board.data[check]) {
                    self->board.data[pos]++;
                    self->board.data[check] = 0;
                    changed = 1;
                    break;
                } else {
                    break;
                }
            }
        }
    }

    return PyBool_FromLong(changed);
}

/**
 * Board_place_tile(self)
 * 
 * Places a new tile on a random empty cell. The value is 1 (90% chance) or 2 (10% chance).
 * Returns True if a tile was placed, False if the board is full.
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

    if (empty_count == 0)
        return PyBool_FromLong(0);  // no empty cell, nothing placed

    // Pick a random empty cell
    int pos = empty[rand() % empty_count];

    // 90% chance of 1, 10% chance of 2
    self->board.data[pos] = (rand() % 10 == 0) ? 2 : 1;

    return PyBool_FromLong(1);  // tile was placed
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
        "to_list",
        (PyCFunction)Board_to_list,
        METH_NOARGS,
        "to_list()\n\n"
        "Return the contents as a Python list of ints."
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