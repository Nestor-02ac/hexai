# cy_board.pxd - Cython declarations for CYBoard
cdef class CYBoard:
    cdef public int size, n, move_count
    cdef int* board
    cdef int* parent1
    cdef int* parent2
    cdef int* rank1
    cdef int* rank2
    cdef int* component_mask1
    cdef int* component_mask2
    cdef bint winner1, winner2
    cdef int* row_starts
    cdef int* row_of_idx
    cdef int* col_of_idx
    cdef int* cell_side_mask
    cdef int* neighbor_count
    cdef int** neighbors
    cdef bint _owns_precomputed

    cdef inline int _find(self, int x, int player) noexcept nogil
    cdef inline void _union(self, int a, int b, int player) noexcept nogil
    cpdef bint play(self, int idx, int player)
    cpdef void play_unchecked(self, int idx, int player)
    cpdef bint check_win(self, int player)
    cpdef list get_empty_cells(self)
    cpdef int get_cell(self, int idx)
    cpdef set_cell(self, int idx, int val)
    cpdef CYBoard clone(self)
    cpdef int rc_to_idx(self, int r, int c)
    cpdef tuple idx_to_rc(self, int idx)
