# chex_board.pxd - Cython declarations for CHexBoard
cdef class CHexBoard:
    cdef public int size, n, move_count
    cdef public int TOP, BOTTOM, LEFT, RIGHT
    cdef int* board
    cdef int* parent
    cdef int* rank
    cdef int* neighbor_count
    cdef int** neighbors
    cdef int* bridge_count
    cdef int** bridge_data
    cdef bint _owns_precomputed

    cdef inline int _find(self, int x) noexcept nogil
    cdef inline void _union(self, int a, int b) noexcept nogil
    cpdef bint play(self, int idx, int player)
    cpdef void play_unchecked(self, int idx, int player)
    cpdef bint check_win(self, int player)
    cpdef list get_empty_cells(self)
    cpdef int get_cell(self, int idx)
    cpdef set_cell(self, int idx, int val)
    cpdef CHexBoard clone(self)

cdef object encode_board_tensor_c(CHexBoard board, int current_player)
