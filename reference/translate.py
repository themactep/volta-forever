#!/usr/bin/env python3
"""translate.py IN.cubin OUT.cubin [MAP]: rewrite sm_75 or supported sm_80 machine code for sm_70.

A prototype of the translator, written to be checked: every decision comes from the descriptions of
the two instruction sets (md.py), from rules that nvdisasm confirmed (remap.py) and from distances
measured in native sm_70 code (latency.py). What it does not know how to do, it refuses, function by
function: a refused function is left out of the result as a trap, and listed.

For an executable sm_80 cubin, ampere.py validates each accepted instruction and the global memory
descriptors on every incoming path. It normalizes the operands before the ordinary translation
steps and requests Volta scheduling for every resulting instruction.

For each function (a `.text` section):
1. every instruction is decoded with the sm_75 description; one that is, bit for bit, an sm_70
   instruction stays as it is;
2. instructions on uniform registers and predicates become their ordinary counterparts (a rule), the
   uniform registers being ordinary registers above those the function uses;
3. the instructions sm_70 lacks become short sequences: the upper half of a product, minimum and
   maximum, addresses with a stride, constants read into uniform registers, memory instructions whose
   address has a uniform part;
4. inserted instructions move the rest: branches and calls, the return addresses that the code
   passes as numbers, and the offsets that the container records are brought to the new places;
5. stall counts are raised where a reader now follows its writer more closely than native sm_70 code
   ever lets it;
6. every instruction of the result must be an instruction of the sm_70 description.

MAP, if given, receives one line per instruction of the result (function, new offset, old offset,
what was done), for validate.py.
"""
import collections, os, re, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elf
import ampere
import machine as MC
import pairing
from machine import ZERO, USCHED, WRITE_BARRIER, READ_BARRIER, WAIT, REUSE, NOP, marked, stall_of, with_stall


class Refused(Exception):
    pass


class Item:
    """One instruction of the result."""
    __slots__ = ('word', 'origin', 'first', 'changed', 'what', 'target', 'fix', 'd', 'late', 'then', 'also', 'says', 'borrows', 'covered', 'more_reads', 'more_writes', 'anew')

    def __init__(self, word, origin, what, changed=True, first=False):
        self.word, self.origin, self.what, self.changed, self.first = word, origin, what, changed, first
        self.late = False       # it reads, some time after it is issued, registers the translation gave it
        self.then = None        # a scoreboard that the instruction after it waits for
        self.also = []          # others it waits for too (borrowed registers: `Function.translate`)
        self.says = False       # an instruction of the program made to say when it has read (the same)
        self.borrows = ()       # the borrowed registers that the sequence it is part of writes for the translation
        self.covered = False    # its result is waited for with that of a later instruction of its sequence
        self.anew = False       # of code for sm_80: the distances that the program had are not those Volta needs
        # What it reads and writes besides what its operands say: the predicates that P2R saves
        # and R2P gives back are in a mask.
        self.more_reads, self.more_writes = [], []
        self.target = None      # the old offset its relative operand points to
        self.fix = None         # for an immediate that is a code offset: the old offset it stands for
        self.d = None


OFFSET_LISTS = {0x1c, 0x1d, 0x25, 0x27, 0x28, 0x2d, 0x31, 0x46}      # attributes that are lists of code offsets
OFFSET_PAIRS = {0x2e: 0, 0x44: 0, 0x55: 1}                           # pairs with a code offset at this place
# (0x10, EIATTR_REQNTID: the block dimensions a kernel asks for, three words, no place in the code;
# Triton's cubins carry it, compilers of CUDA C do not.)
PLAIN = {0x04, 0x05, 0x0a, 0x0f, 0x10, 0x11, 0x12, 0x17, 0x19, 0x1b, 0x1e, 0x23, 0x29, 0x2f, 0x33, 0x36, 0x37, 0x45, 0x4c, 0x5f, 0x66}
STRIDES = {1: 2, 2: 3, 3: 4}                                         # the stride field: X4, X8, X16 as shifts
UNIFORM_ADDRESS, SHARED, STRIDED = MC.UNIFORM_ADDRESS, MC.SHARED, MC.STRIDED


LATE_WRITERS = ('MIO_FAST', 'MIO_SLOW')   # the kinds whose results are written when they arrive
DEEP_WRITERS = ('DADD', 'DMUL', 'DFMA', 'HMMA')     # fixed-time arithmetic whose result is written two cycles after the others'


def unannounced(d):
    """The instruction `d` writes its registers later than an instruction issued a cycle or two
    after it would, and names no scoreboard that says when (`Borrowed`)."""
    return d.mnemonic in DEEP_WRITERS or (d.kind in LATE_WRITERS and (d.word >> WRITE_BARRIER) & 7 == 7)
ARRIVES = 4                        # the scoreboard on which a load added by the translation says its result is there
SEQUENCE = 5                       # the scoreboard of the instructions of a sequence that stands for a matrix instruction
MATRIX = ('ldsm__sImmOffset', 'ldsm__URsIR', 'hmma_x8_')      # Turing's matrix load and product (`Function.matrix_load`, `matrix_product`)
MATRIX_SCRATCH = 8                 # the registers that the sequence of a matrix instruction computes into
KEPT = 4                           # the registers of what a function with a product keeps of the lane (`Function.kept`)
GATHERED = 4                       # the registers that keep the elements of B a product gathered (`Function.matrix_product`)
AROUND = ('wait', 'save', 'restore', 'lane', 'kept', 'settled', 'to pairs', 'from pairs')   # what the translation puts around an instruction, which is not what the instruction became
# Pairs of products on whole rows (pairing.py; docs/pairs.md): the values kept of the lane
# for them, in the order of their registers; whether the rows of a matrix load are taken to be
# aligned on 16 bytes (the policy of the program-text lowering: wide loads then); whether any of it is done.
ROW_KEPT = ('kA', 'kA2', 'kB', 'kB2', 'kBT', 'cB', 'pi', 'inv')
ALIGNED_ROWS = os.environ.get('VF_ALIGNED_ROWS', '1') != '0'
PAIRS = os.environ.get('VF_PAIRS', '1') != '0'       # (VF_PAIRS=0: every product by itself, as before 2026-11-18)
# The relocations that write the lower and the upper half of an address into the 32 bits of an
# immediate (R_CUDA_ABS32_LO_32, R_CUDA_ABS32_HI_32).
HALVES = (56, 57)
# What ptxas 12.9.2 writes for `trap` on sm_70: BPT.TRAP 0x1, then a branch to itself.
TRAP, STAY = 0x000fea0000300000000000040000795c, 0x000fc0000383fffffffffff000007947


def ordinary_back(operand, ops):
    """The name, among the operands `ops` of an sm_75 instruction, of the uniform operand that became `operand`."""
    for name in ('U' + operand, operand.replace('_R', '_UR')):
        if name in ops:
            return name
    return operand


def relative(mc, sm, d):
    """(field, scale) of the operand of an instruction that is a distance in the code, or None."""
    names = [n for n, o in d.c.operands.items() if o.type == 'RSImm']
    if d.c.name == 'brx_':
        # A branch through a register goes to the instruction after it, plus the register, plus a
        # number. The register holds an offset in the function, read from a table (`tables_of`);
        # the number is minus where the next instruction is: a distance in the code, to offset 0.
        names = ['Ra_offset']
    if not names:
        return None
    dec = mc.d75 if sm == 75 else mc.d70
    for step, fields, data in dec.plan(d.c):
        if step == 'scaled' and data[0] == names[0]:
            return fields[0], data[1]
    raise Refused('a relative operand of %s that is not understood' % d.c.name)


def signed(field, word):
    v = field.get(word)
    return v - (1 << len(field)) if v >> (len(field) - 1) else v


def graph_register_budget(registers, info):
    """Largest register claim that retains all source launch block sizes on GV100."""
    max_threads = 1024
    pos = 0
    while pos < len(info):
        if pos + 4 > len(info):
            max_threads = 1024
            break
        fmt, kind, size = struct.unpack_from('<BBH', info, pos)
        size = size if fmt == 4 else 0
        if pos + 4 + size > len(info):
            max_threads = 1024
            break
        if kind in (0x05, 0x10):
            if fmt != 4 or size != 12:
                max_threads = 1024
                break
            x, y, z = struct.unpack_from('<3I', info, pos + 4)
            if not 0 < x * y * z <= 0xffffffff:
                max_threads = 1024
                break
            max_threads = min(max_threads, x * y * z)
        pos += 4 + size
    per_warp = ((32 * max(registers, 1) + 255) // 256) * 256
    threads = min(max_threads, (16384 // per_warp) * 4 * 32)
    return block_registers(threads)


def block_registers(threads):
    """The most registers a thread may have for a block of `threads` threads to be launched on a
    GV100 (the admission check counts warps by fours: each 128 threads take the same)."""
    groups = max(1, (threads + 127) // 128)
    return min(255, (65536 // (128 * groups) // 8) * 8)


def required_threads(info):
    """The block size an entry requires (EIATTR_REQNTID, which Triton writes for every kernel and
    CUDA C never): the driver launches it with that block and no other. None when it requires
    none, or one that is not understood or that no GPU launches (more than 1,024 threads)."""
    required, pos = None, 0
    while pos < len(info):
        if pos + 4 > len(info):
            return None
        fmt, kind, size = struct.unpack_from('<BBH', info, pos)
        size = size if fmt == 4 else 0
        if pos + 4 + size > len(info):
            return None
        if kind == 0x10:
            if fmt != 4 or size != 12:
                return None
            x, y, z = struct.unpack_from('<3I', info, pos + 4)
            if not 0 < x * y * z <= 1024:
                return None
            required = x * y * z if required is None else min(required, x * y * z)
        pos += 4 + size
    return required


def tables_of(data):
    """{offset of a branch through a register: the offsets it may lead to}, from the attributes of a
    function: the record of its jump tables (0x34) is, for each, the branch, a zero, how many
    places, the places."""
    out, pos = {}, 0
    while pos + 4 <= len(data):
        fmt, attr, val = data[pos], data[pos + 1], struct.unpack_from('<H', data, pos + 2)[0]
        if fmt == 4 and attr == 0x34:
            whole = max(0, (min(pos + 4 + val, len(data)) - (pos + 4)) // 4)
            words = struct.unpack_from('<%dI' % whole, data, pos + 4)
            at = 0
            while at < len(words):
                if at + 3 > len(words) or words[at + 1] != 0 or at + 3 + words[at + 2] > len(words):
                    raise Refused('a record of jump tables that is not understood')
                out[words[at]] = list(words[at + 3:at + 3 + words[at + 2]])
                at += 3 + words[at + 2]
        pos += 4 + val if fmt == 4 else 4
    return out


# Whether the translation looks for room in the program's own registers before it takes new ones
# (`Borrowed`). 0: off. 1: the homes and the pool (levels 2, 1 and 0 of `attempt`), which the Rust
# translator has too. 2: and, for a kernel that is refused even so, everything at the instruction
# (level 3). 3: level 3 first, with every uniform register in memory, for tests.
# 4: by the registers a kernel may have without losing a thread (its budget): the first of levels
# 0, 1, 2 and 3 whose translation stays within it (an experiment).
# 5: the same, and at level 3 room is made where there is none (a register the instruction does
# not touch is put in the frame for the time of its sequence), the frame being addressed from
# the stack pointer where the function never moves it (an experiment too).
# 6: exactly as with the switch at 0 wherever that translates, and for a kernel refused for want
# of registers, levels 1, 2, then 3 with room made. What the layer did by itself
# until 2026-11-20.
# 7: what the layer does by itself, and the default: what 6 does, and a function with matrix
# products may be translated at level 3: the lane is read there, the four values computed
# there, B is gathered every time, and every register the product needs is taken at the
# instruction (docs/translator.md, "Registers").
# 8: level 3 first for EVERY kernel, products included, for tests (as 3 is today without products).
BORROW_DEFAULT = 7
BORROW = int(os.environ.get('VF_BORROW') or BORROW_DEFAULT)   # (unset or empty: the default; 0: off)


def stack_top(d):
    """Whether instruction `d` reads c[0x0][0x28] as an operand: what the stack pointer of a
    thread starts at, the top of its local memory."""
    return any(key.endswith('_bank') and value == 0 and d.ops.get(key[:-5] + '_addr') == 0x28 // 4 for key, value in d.ops.items())


class Borrowed:
    """Registers of the program that hold nothing where the translation needs registers: a home
    for each uniform register through its whole life, and an even register and the next at an
    instruction whose sequence computes into two (the pool). docs/translator.md, "Registers".

    A register is free at an instruction when it is dead before it, dead after it, and neither
    read nor written by it (an instruction may write a register that nothing reads).
    The lives follow the ways of `pairing.Flow`; a write
    under a predicate ends no life; a call within the section reads and writes what the code
    it calls does; a call elsewhere reads every register the function names; at a return every
    one of them is alive. R0 and R1 are never taken (R1 is the stack).

    Nor is a register that some instruction of the function writes late without saying when:
    a conversion, a load from shared memory, MUFU, POPC under no scoreboard (the compiler counted
    the cycles to their readers, 7 to 20 and more; measured, their result is written 9 to 11
    cycles after they are issued, over what was written meanwhile), and the arithmetic on
    doubles and Volta's matrix product, whose results are written two cycles later than those
    of every other arithmetic (a MOV issued one or two cycles after a DMUL is written over by
    it). Where nothing reads such a result on some way, the register holds nothing while the
    result is still to come, and no wait can be put on it (`unannounced`)."""

    def __init__(self, f, all_in_memory=False, top=None):
        flow = pairing.Flow(f)
        if not flow.known:
            raise Refused('the ways through the function are not known')
        src, n = f.src, len(f.src)
        def bits(entries):
            out = 0
            for entry in entries:
                letter, x = entry[0], entry[1]
                if letter == 'R' and x != 255:
                    out |= 1 << x
                elif letter == 'U' and x != 63:
                    out |= 1 << (256 + x)
            return out
        reads, writes = [bits(d.reads) for d in src], [bits(d.writes) for d in src]
        every = 0
        for i in range(n):
            every |= reads[i] | writes[i]
        kills, leaves = list(writes), [0] * n
        for i in range(n):
            if flow.kind[i] == 'call':
                called = flow.called.get(i)
                if called is None:
                    reads[i] |= every
                else:
                    reads[i] |= sum(1 << x for x in called[0])
                    writes[i] |= sum(1 << x for x in called[1])
                    # (And the uniform registers that the code it calls reads or writes:
                    # `flow.called` has the registers only. A uniform register set before the
                    # call and read by that code was taken for dead at the call until 2026-11-19.)
                    body, work = set(), [flow.named[i]]
                    while work:
                        k = work.pop()
                        if k not in body:
                            body.add(k)
                            work += flow.succ[k]
                    for k in body:
                        reads[i] |= bits(src[k].reads) >> 256 << 256
                        writes[i] |= bits(src[k].writes) >> 256 << 256
                kills[i] = 0
            elif not pairing.unguarded(src[i]):
                kills[i] = 0
            if src[i].mnemonic == 'RET':
                leaves[i] = every
        live_in, changed = [0] * n, True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                out = leaves[i]
                for t in flow.succ[i]:
                    out |= live_in[t]
                new = reads[i] | (out & ~kills[i])
                if new != live_in[i]:
                    live_in[i], changed = new, True
        busy = []
        for i in range(n):
            out = leaves[i]
            for t in flow.succ[i]:
                out |= live_in[t]
            busy.append(live_in[i] | out | reads[i] | writes[i])
        never = 0
        for d in src:
            if unannounced(d):
                never |= bits(d.writes)
        # (`top`: the highest register that may be taken. The function's own highest; or, with
        # a budget, registers above it too, which hold nothing anywhere.)
        self.top = f.highest if top is None else max(top, f.highest)
        self.busy = [(b | never) & ((1 << 255) - 1) for b in busy]
        # To make room (`victims_at`): what each instruction reads or writes itself, the
        # registers that receive a result nothing announces, the function's highest register.
        self.touch = [(reads[i] | writes[i]) & ((1 << 255) - 1) for i in range(n)]
        self.never, self.highest = never & ((1 << 255) - 1), f.highest
        # And where the stack pointer is, for a frame addressed from it: `depth[i]`, how far R1
        # is, before instruction i, under where the first instruction puts it. The program
        # moves it by constants only (PyTorch: 730 functions of 23,693 move it at all, every
        # one by `IADD3 R1, R1, imm`): known at an instruction when every way to it agrees;
        # code that is called has the depth of its calls. `r1_fixed`: known wherever the first
        # instruction leads; otherwise the frame is addressed from a register (`frame_top`).
        self.depth, self.r1_fixed = [None] * n, n > 1
        work = [(t, 0) for t in flow.succ[0]] if n > 1 else []
        while work and self.r1_fixed:
            i, at = work.pop()
            if self.depth[i] is not None:
                self.r1_fixed = self.depth[i] == at
                continue
            self.depth[i] = at
            d, after = src[i], at
            if writes[i] >> 1 & 1:
                ops = d.ops
                if (d.c.name == 'iadd3_imm__RsIR_RIR' and ops.get('Rd') == 1 and ops.get('Ra') == 1 and ops.get('Rc', 255) == 255
                        and pairing.unguarded(d) and not any(ops.get(k) for k in ('Ra@negate', 'Rc@negate'))):
                    imm = ops['Sb'] - (1 << 32) if ops['Sb'] >> 31 else ops['Sb']
                    after = at - imm
                else:
                    self.r1_fixed = False
            work += [(t, after) for t in flow.succ[i]]
            if flow.kind[i] == 'call' and flow.called.get(i) is not None:
                work.append((flow.named[i], at))
        self.room, self.from_r1, self.spill = False, False, 0
        # The uniform registers, those of the longest lives first: each in the lowest register
        # that is free wherever it is alive, read or written, and that no other one has there.
        self.taken, self.homes, self.used = [0] * n, {}, set()
        self.spans = {}                              # {homed uniform register: the instructions of its life}
        units = []
        for u in f.uniform:
            if u - 1 in f.pairs:
                continue
            width = 2 if u in f.pairs else 1
            mask = (1 << width) - 1
            units.append((u, width, [i for i in range(n) if busy[i] >> (256 + u) & mask]))
        if not all_in_memory:
            for u, width, span in sorted(units, key=lambda unit: (-len(unit[2]), unit[0])):
                union, mask = 0, (1 << width) - 1
                for i in span:
                    union |= self.busy[i] | self.taken[i]
                for r in range(2, self.top + 2 - width, width):
                    if not union >> r & mask:
                        for k in range(width):
                            self.homes[u + k] = r + k
                            self.used.add(r + k)
                            self.spans[u + k] = span
                        for i in span:
                            self.taken[i] |= mask << r
                        break
        # Level 3 (`everything`): the uniform registers that found no home live in the thread's
        # local memory, in a frame the translation adds, and whatever else the translation of an
        # instruction needs is taken at that instruction (`Function.take`).
        self.everything = False
        self.slots = {}           # {uniform register: its word in the frame}
        self.predicate_slots = {} # {held uniform predicate: its word in the frame}
        self.frame_bytes = 0      # the frame: 4 bytes a register, a multiple of 16

    def free_at(self, i, width=1, given=0):
        """The lowest register that holds nothing at instruction `i`, or the lowest run of `width`
        registers from an even one: no home of a uniform register alive there, and not among
        `given` (a mask: what the same instruction was given already). R0 and R1 never, and
        nothing above the registers the function names. None when there is none."""
        held, mask = self.busy[i] | self.taken[i] | given, (1 << width) - 1
        for r in range(2, self.top + 2 - width, 1 if width == 1 else 2):
            if not held >> r & mask:
                return r
        return None

    def victims_at(self, i, width, given=0):
        """Where no register holds nothing: the lowest register, or run of `width` from an even
        one, that instruction `i` neither reads nor writes itself, that is no home there, was
        not given already, and receives no result that nothing announces: registers that hold
        something for the program, to be put in the frame for the time of the sequence and
        brought back after it (`Function.make_room`). None when there is none."""
        held, mask = self.touch[i] | self.taken[i] | given | self.never | 3, (1 << width) - 1
        for r in range(2, self.highest + 2 - width, 1 if width == 1 else 2):
            if not held >> r & mask:
                return r
        return None

    def kept(self, span):
        """A register that holds nothing at any instruction of `span` and is no home there:
        taken through all of it (what holds a uniform predicate between the instructions that
        name it, at level 3). None when there is none."""
        union = 0
        for i in span:
            union |= self.busy[i] | self.taken[i]
        for r in range(2, self.top + 1):
            if not union >> r & 1:
                for i in span:
                    self.taken[i] |= 1 << r
                self.used.add(r)
                return r
        return None

    def assign_slots(self, uniform, pairs):
        """A word of the frame for each uniform register that has no home (the two of a pair
        side by side, the lower one first). Returns the size of the frame in bytes."""
        slot = 0
        for u in sorted(uniform):
            if u in self.homes or u - 1 in pairs:
                continue
            if u in pairs:
                # Pair: 8 bytes (low word first)
                self.slots[u] = slot
                self.slots[u + 1] = slot + 1
                slot += 2
            else:
                self.slots[u] = slot
                slot += 1
        self.frame_bytes = 4 * slot
        # Round up to the same alignment the compiler uses (16 bytes).
        if self.frame_bytes % 16:
            self.frame_bytes = (self.frame_bytes + 15) & ~15
        return self.frame_bytes

    def slot_count(self):
        return len(self.slots) + len(self.predicate_slots)

    def slot_offset(self, word):
        """Where frame word `word` is, from the top of the thread's local memory
        (c[0x0][0x28], what the stack pointer starts at): the frame is right under it."""
        return -self.frame_bytes + 4 * word


def placement(base, uniform, pairs, pool, saves, lane=False, scratch=0, product=False, rows=None, row_kept=(), companions=None, held=()):
    """Where the uniform registers of a function go, or of functions that call one another: ordinary
    registers from `base` on, the first one that none of them uses. `uniform`: the registers, in
    order; `pairs`: the lower one of each two that an instruction takes as one value of 64 bits;
    `pool`: whether two registers are wanted for what the translation computes (`Function.pool`);
    `saves`: how many registers are wanted to save predicates in (`Function.stand_ins`); `lane`:
    whether one is wanted for the number of the thread in its warp; `scratch`: how many for the
    sequences that stand for matrix instructions (`Function.matrix_load`); `product`: whether
    one of those is a product, which has four registers for what is kept of the lane
    (`Function.kept`) and four for the elements of B that a product gathered
    (`Function.matrix_product`).

    {uniform register: ordinary register}, the first of the two registers of the pool (or None),
    the first of the registers to save predicates in (or None), the register of the lane (or
    None), the first of the registers of what is kept of the lane (or None), the first of the
    registers of a gathered B (an even one, or None), the first of the scratch registers (an
    even one, or None) and the first register left free.
    Two registers that are one value stay side by side, the lower one even: all native code of
    both architectures has them so (3.6 million operands, none otherwise). The others fill what
    that leaves: at most one register is lost. (The lane takes the odd register before the ones
    that start on an even one, when there is one: a product then has the seventeen registers it
    had when each product computed everything again.)"""
    if any(u + 1 in pairs for u in pairs):
        raise Refused('uniform registers in pairs that overlap')
    singles = [u for u in uniform if u not in pairs and u - 1 not in pairs]
    place, free, where = {}, base, None
    aligned = sorted(pairs) + ([None] if pool else [])
    if aligned and free % 2:
        if singles:
            place[singles.pop(0)] = free
        free += 1
    for u in aligned:
        if u is None:
            where = free
        else:
            place[u], place[u + 1] = free, free + 1
        free += 2
    for u in singles:
        place[u] = free
        free += 1
    save, free = (free if saves else None), free + saves
    # (`held`: the uniform predicates that some life holds in a register: one each.)
    hold_at = {}
    for q in held:
        hold_at[q], free = free, free + 1
    lane_at = kept_at = gathered_at = scratch_at = None
    if lane and free % 2:
        lane_at, free = free, free + 1
    if product:
        free += free % 2
        gathered_at, free = free, free + GATHERED
    if scratch:
        free += free % 2
        scratch_at, free = free, free + scratch
    if lane and lane_at is None:
        lane_at, free = free, free + 1
    if product:
        kept_at, free = free, free + KEPT
    # (`rows`: {destination of whole-row loads: 4 or 8 registers}; `row_kept`: the values kept
    # of the lane for them. Both after everything else: a function without pairs has the
    # registers it had.)
    kept_rows = {}
    for name in row_kept:
        kept_rows[name], free = free, free + 1
    rows_at = {}
    for rd in sorted(rows or {}):
        free += free % 2
        rows_at[rd], free = free, free + rows[rd]
    # (`companions`: {root register: how many registers its companions take}, last of all.)
    comp_at = {}
    for root in sorted(companions or {}):
        comp_at[root], free = free, free + companions[root]
    if free > 253:
        # The register count is two more than the registers used, and has eight bits.
        if os.environ.get('VF_PAIRS_SAY'):
            print('  %d registers wanted: the program\'s %d, uniform %d, pool %d, saves %d, lane %d, gathered and kept %d, scratch %d, kept for rows %d, rows %s'
                  % (free, base, len(uniform), 2 * bool(pool), saves, bool(lane), (GATHERED + KEPT) * bool(product), scratch, len(row_kept), dict(rows or {})), file=sys.stderr)
        raise Refused('more registers than there are')
    return place, where, save, lane_at, kept_at, gathered_at, scratch_at, free, rows_at, kept_rows, comp_at, hold_at


class Function:
    def __init__(self, mc, name, code, registers, tables=None):
        self.mc, self.name, self.registers = mc, name, registers
        self.ampere = False
        self.ampere_ops = {}
        self.tables = tables or {}
        if len(code) % 16:
            raise Refused('code that is not a whole number of instructions')
        self.words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code), 16)]
        self.src = [mc.decode(75, w) for w in self.words]
        for i, d in enumerate(self.src):
            if d is None:
                raise Refused('the instruction %032x at 0x%x is not one of the sm_75 description' % (self.words[i], 16 * i))
        # The uniform registers it reads or writes; the lower one of each two that an instruction
        # takes as one value (a constant of 64 bits, an address, a product); and whether any of
        # its instructions needs registers to compute into (`pool`).
        uniform, self.pairs, self.needs_pool = set(), set(), False
        for d in self.src:
            uses = mc._uses[(75, d.c.name)]
            for letter, operand, span in uses.reads + uses.writes:
                if letter == 'U' and operand in d.ops and d.ops[operand] != 63:
                    count = MC.registers(span, d.ops)
                    uniform.update(range(d.ops[operand], d.ops[operand] + count))
                    if count == 2:
                        self.pairs.add(d.ops[operand])
                    elif count != 1:
                        raise Refused('a uniform operand of %d registers' % count)
            name = d.c.name
            if (name.startswith('imad_hi__') or (name in STRIDED and d.ops.get('stride'))
                    or (name.startswith(tuple(UNIFORM_ADDRESS)) and d.ops.get('Ra', 255) != 255)):
                self.needs_pool = True
        # Turing's matrix instructions become sequences that go by the lane of the thread, read
        # once when the function is entered, and compute into registers of their own. With a
        # product, four values made of the lane are computed there too (`kept`), and four
        # registers keep the elements of B that the last product gathered (`matrix_product`).
        self.needs_lane = any(d.c.name in MATRIX for d in self.src)
        self.has_product = any(d.c.name == 'hmma_x8_' for d in self.src)
        self.product_count = sum(1 for d in self.src if d.c.name == 'hmma_x8_')
        self.scratch_need = MATRIX_SCRATCH if self.needs_lane else 0
        self.q = None                               # the plan of pairs on whole rows (`plan_pairs`)
        self.borrow = None                          # the registers of its own that the translation may take (`Borrowed`)
        self.entered_inside = False                 # set by `translate`: a symbol of another function points into it
        self.uniform = sorted(uniform - {63})
        self.uniform_predicates = sorted({r for d in self.src for l, r, *_ in d.reads + [w + (None,) for w in d.writes] if l == 'Q'})
        self.predicates = {r for d in self.src for l, r, *_ in d.reads + [w + (None,) for w in d.writes] if l == 'P'}
        self.highest = max([r for d in self.src for l, r, *_ in d.reads + [w + (None,) for w in d.writes] if l == 'R'] or [0])

    def plan_pairs(self, wanted=True, companions=True):
        """Decide which products are done as pairs on whole rows (pairing.py), or none of them;
        and whether their loads may take addresses from companions."""
        self.q = pairing.plan(self, companions) if wanted and PAIRS and self.needs_lane and not self.entered_inside else None
        sites = self.q.sites if self.q else {}
        # (A product that is not half of a pair is done as before, with what that keeps and gathers.)
        self.has_product = any(d.c.name == 'hmma_x8_' and i not in sites for i, d in enumerate(self.src))

    # ---- translation -------------------------------------------------------------------------------------
    def stand_ins(self):
        """The ordinary predicate that stands for each uniform predicate at each instruction where
        the uniform one holds something (`stand`), and a predicate to compute into for each
        minimum and maximum (`scratch_at`); `save_slots`: how many registers that needs to save
        predicates in.

        A uniform predicate has separate lives: a carry is born and dies within two instructions,
        many times in a function. Each life gets an ordinary predicate that holds nothing of the
        program all along it and stands for no other uniform predicate there.

        Where there is none (all seven hold something), the program has one that it neither reads
        nor writes during that life: its value is saved in a register before the life (P2R) and
        given back after it (R2P), as native code does when it runs out of predicates, and it
        stands for the uniform one in between. That takes a life that is a run of instructions in
        a straight line: entered at its first, left after its last.

        Where that cannot be either (the program touches all seven during the life, or the life
        goes through branches), the uniform predicate is **held** in a register of the
        translation for that life: written there after each instruction that writes it (zero
        or not zero), and read back into an ordinary predicate before each instruction that
        reads it. The ordinary predicate is then needed at those instructions only: one that
        holds nothing there, or one the instruction does not touch, saved around it.
        (`held`: {instruction: [(uniform predicate, ordinary predicate, read back before,
        written after)]}; `held_lives`: [(uniform predicate, the instructions of the life)].)
        Until 2026-11-18 such a function was refused: 133 of PyTorch's, f32 attention among them.

        **A predicate that some instruction of the function writes late is the last one taken,
        and taken with a delay** (`slow`, `delayed`). A comparison of doubles writes its
        predicate two cycles later than the others do, FCHK 13 to 15 cycles after it is issued,
        with a scoreboard or without (measured on a V100): what the translation puts in the
        predicate meanwhile is written over, or its save reads what the predicate held before.
        The schedule knows the writers of its own block only, so it cannot keep the distance
        where the lanes come by a jump. So a predicate that no such instruction writes is taken
        wherever there is one; where there is none, the instruction whose translation writes or
        saves the slow one begins by instructions that do nothing: the first waits for every
        scoreboard, and together they last as long as native code keeps a plain reader from
        such a writer under no scoreboard (13 cycles after DSETP, 26 after FCHK: the compiler's
        own distances; measured, a P2R reads what a DSETP wrote from 8 cycles on, and an ISETP
        issued 15 cycles after an FCHK is the one that stays). Whatever was issued before has
        written by then. One that a memory instruction writes under no scoreboard is never
        taken: no count of cycles is enough.
        (`delayed`: {instruction: cycles}. Until 2026-11-19 nothing looked at this; 492 of
        PyTorch's functions took such a predicate.)"""
        self.liveness()
        n = len(self.src)
        never, slow, wait = 0, 0, [0] * 7
        for d in self.src:
            if d.mnemonic == 'DSETP' or d.kind in LATE_WRITERS:
                for letter, register in d.writes:
                    if letter != 'P':
                        continue
                    slow |= 1 << register
                    if (d.word >> WRITE_BARRIER) & 7 != 7:
                        wait[register] = max(wait[register], 1)         # (on a scoreboard: waited for)
                    elif d.kind == 'MIO_SLOW':
                        never |= 1 << register
                    else:
                        wait[register] = max(wait[register], self.mc.least('P', 'data', d, 'FXU'))
        self.slow_predicates, self.delayed = slow, {}
        def pick(taken, order):
            """The first predicate of `order` that is not in `taken`: one that nothing writes
            late, or else one whose late writers can be waited for."""
            for avoid in (slow, never):
                p = next((p for p in order if not (taken | avoid) >> p & 1), None)
                if p is not None:
                    return p
            return None
        def delay(i, p):
            # (The translation writes or saves predicate p at instruction i.)
            if slow >> p & 1:
                self.delayed[i] = max(self.delayed.get(i, 0), wait[p])
        busy = [self.live_in[i] | self.live_out[i] | self.defs[i] | self.uses[i] for i in range(n)]
        self.stand, self.taken_at, self.lives = {}, [0] * n, []
        self.held, self.held_lives = collections.defaultdict(list), []
        self.saves, self.restores = collections.defaultdict(list), collections.defaultdict(list)
        runs = []                                    # (first, last, slot) of what is saved
        def slot(i0, i1):
            used = {k for a, b, k in runs if a <= i1 and i0 <= b}
            k = next(k for k in range(len(runs) + 1) if k not in used)
            runs.append((i0, i1, k))
            return k
        def give(q, part, p, saved):
            for i in part:
                self.stand[(q, i)] = p
                self.taken_at[i] |= 1 << p
            self.lives.append((q, p, saved, part))
        lives = []
        for q in self.uniform_predicates:
            # A life: instructions where it holds something, joined where the value goes from one
            # to the next.
            alive = [i for i in range(n) if busy[i] >> (7 + q) & 1]
            parent = {i: i for i in alive}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for i in alive:
                for j in self.successors[i]:
                    if j in parent and self.live_out[i] >> (7 + q) & 1 and self.live_in[j] >> (7 + q) & 1:
                        parent[find(i)] = find(j)
            parts = collections.defaultdict(list)
            for i in alive:
                parts[find(i)].append(i)
            lives += [(part[0], q, part) for part in parts.values()]
        hard = []
        for first, q, part in sorted(lives):
            if any(self.external[i] for i in part):
                raise Refused('a uniform predicate is alive across a call to another function')
            taken = 0
            for i in part:
                taken |= (busy[i] & 0x7f) | self.taken_at[i]
            p = pick(taken, range(7))
            if p is None:
                hard.append((q, part))
            else:
                give(q, part, p, False)
                for i in part:
                    if self.defs[i] >> (7 + q) & 1:
                        delay(i, p)
        targets = {j for i in range(n) for j in self.successors[i] if j != i + 1}
        def straight(i):
            d = self.src[i]
            return self.successors[i] == [i + 1] and not self.external[i] and (d.kind != 'BRU' or d.mnemonic in ('BSSY', 'BMOV'))
        later = []
        for q, part in hard:
            i0, i1 = part[0], part[-1]
            touched = 0
            for i in range(i0, i1 + 1):
                touched |= ((self.defs[i] | self.uses[i]) & 0x7f) | self.taken_at[i]
            p = pick(touched, range(7))
            if (p is None or part != list(range(i0, i1 + 1)) or any(i in targets for i in range(i0 + 1, i1 + 1))
                    or not all(straight(i) for i in range(i0, i1 + 1))):
                later.append((q, part))
                continue
            k = slot(i0, i1)
            give(q, part, p, True)
            self.saves[i0].append((k, p))
            self.restores[i1].append((k, p))
            delay(i0, p)
        # The lives that are held in a register: a predicate for each instruction that names the
        # uniform one, there only. (After the lives above: what stands for those all along them
        # is taken.)
        for q, part in later:
            for i in part:
                reads, writes = self.uses[i] >> (7 + q) & 1, self.defs[i] >> (7 + q) & 1
                if not (reads or writes):
                    continue
                taken = (busy[i] & 0x7f) | self.taken_at[i]
                p = pick(taken, range(6, -1, -1))
                saved = p is None
                if saved:
                    touched = ((self.defs[i] | self.uses[i]) & 0x7f) | self.taken_at[i]
                    p = pick(touched, range(6, -1, -1))
                    # (Given back after the instruction: which must then go on to the next one, always.)
                    if p is None or not straight(i):
                        raise Refused('no predicate is free while a uniform predicate is alive')
                    k = slot(i, i)
                    self.saves[i].append((k, p))
                    self.restores[i].append((k, p))
                delay(i, p)
                self.stand[(q, i)] = p
                self.taken_at[i] |= 1 << p
                # (An instruction that may leave the uniform predicate as it was, under a guard
                # that does not hold, finds it in the ordinary one: read back before it too.)
                self.held[i].append((q, p, bool(reads or (writes and not self.kills[i] >> (7 + q) & 1)), bool(writes)))
                self.lives.append((q, p, saved, [i]))
            self.held_lives.append((q, part))
        self.held_predicates = sorted({q for q, part in self.held_lives})
        self.scratch_at = {}
        for i, d in enumerate(self.src):
            if d.c.name.startswith('imnmx__'):
                # One that holds nothing alive after it, is not one of its own, and does not stand
                # for a uniform predicate alive there; or else one it does not touch, saved.
                taken = (busy[i] & 0x7f) | self.taken_at[i]
                p = pick(taken, range(6, -1, -1))
                if p is not None:
                    self.scratch_at[i] = (p, None)
                else:
                    touched = (self.defs[i] | self.uses[i]) & 0x7f
                    for q in self.uniform_predicates:
                        if (self.defs[i] | self.uses[i]) >> (7 + q) & 1:
                            touched |= 1 << self.stand[(q, i)]
                    p = pick(touched, range(6, -1, -1))
                    if p is None:
                        raise Refused('no predicate to compute a minimum into at 0x%x' % (16 * i))
                    self.scratch_at[i] = (p, slot(i, i))
                delay(i, p)
        self.save_slots = 1 + max((k for a, b, k in runs), default=-1)

    def translate(self, place, pool, save, lane, kept, gathered, scratch, free, rows_at=None, kept_rows=None, comp_at=None, hold_at=None):
        """`place`, `pool`, `save`, `lane`, `kept`, `gathered`, `scratch`, `free`: where the
        uniform registers go, the two registers to compute into, the first of the registers to
        save predicates in, the register of the lane, the first of the registers of what is kept
        of the lane, the first of those of a gathered B, the first of the scratch registers and
        the first register left free, for this function and those it calls or that call it
        (`placement`)."""
        mc = self.mc
        self.place, self.pool_at, self.save_at, self.next_register = place, pool, save, free
        self.pool_taken = {}                        # the instructions whose sequence took borrowed registers: the first of them
        self.pool_width = {}                        # and how many
        # Level 3, everything at the instruction: what the instruction at hand was given (a
        # mask), every run given at each instruction (for the validator: `write_lives`), the
        # registers of the lane and of the scratch of a matrix load, those that hold the uniform
        # registers of the frame, and the one that holds the top of the frame.
        self.given, self.given_at = 0, {}
        self.spilled, self._lower, self._top_again = {}, None, None       # (room made: `make_room`)
        self._lane_taken, self._scratch_taken = {}, {}
        self._mem_regs, self._base = {}, None
        self.lane_at, self.matrix_at = lane, scratch      # (`scratch_at` is the predicates of minima: `stand_ins`)
        self.kept_at, self.gathered_at = kept, gathered
        self.rows_at, self.kept_rows, self.comp_at, self.hold_at = rows_at or {}, kept_rows or {}, comp_at or {}, hold_at or {}
        self.rows_n = self.q.rows() if self.q else {}
        # The product whose gathering of B the registers kept for that hold: its index and its
        # register of B (`kept_gathering`); the instructions that are not only entered from the
        # one before them (`entries`).
        self.holds, self.entered = None, self.entries() if self.has_product else None
        if self.needs_lane and self.entered_inside:
            raise Refused('matrix instructions in a function that can be entered elsewhere than at its start')
        def R(u):
            if u == 63:
                return 255
            if u in self._mem_regs:
                return self._mem_regs[u]
            if u not in place:
                raise Refused('uniform register UR%d is used in a way that was not foreseen' % u)
            return place[u]
        def P(q):
            if q != 7 and (q, self.at) not in self.stand:
                raise Refused('uniform predicate UP%d is used in a way that was not foreseen' % q)
            return 7 if q == 7 else self.stand[(q, self.at)]
        self.R, self.P = R, P
        items = []
        targets = set()
        for i, d in enumerate(self.src):
            self.at = i
            self.given, self._mem_regs, self._base = 0, {}, None
            frame = self.borrow.frame_bytes if self.borrow is not None else 0
            if i == 0:
                self._entry_sets_r1 = stack_top(d) and d.writes == [('R', 1)] and pairing.unguarded(d)
            elif stack_top(d) and self._top_again is None:
                self._top_again = i
            if frame and (stack_top(d) if i else not (stack_top(d) and d.writes == [('R', 1)] and pairing.unguarded(d))):
                # The frame is the top of the thread's local memory, and the program's own frame
                # is kept under it by lowering the stack pointer where it is set: at the first
                # instruction of a kernel, once.
                raise Refused('uniform registers in memory, and the stack pointer is not set once, by the first instruction' if i == 0
                              else 'uniform registers in memory, and the top of the stack is read again at 0x%x' % (16 * i))
            # Level 3: the uniform registers of the frame that the instruction reads are loaded
            # before it, those it writes are stored after it.
            before = self.from_memory(i, d)
            operation = self.ampere_ops.get(i)
            if isinstance(operation, tuple) and operation[0] == 'increment':
                if self.matrix_at is None:
                    raise Refused('no dedicated registers for an sm_80 shared atomic increment')
                scratch = self.matrix_at
                addr = self.mc.assemble('lea_lo_imm__RuIR_RIR', Rd=scratch, Ra=d.ops['Ra'], Sb=0, scaleU5=2, Pu=7)
                one = self.mc.assemble('imad__RRsI_RRI', Rd=scratch+1, Ra=255, Rb=255, Sc=1)
                atom = self.mc.assemble('atoms__RaNonRZ', Ra=scratch, Ra_offset=operation[1], Rb=scratch+1, Rd=255,
                                        op='ADD', sz='32', Pg=d.ops['Pg'], Pg_not=d.ops['Pg@not'], src_rel_sb=SEQUENCE)
                new = [Item(addr | 63 << WAIT, i, 'sm80: index shared counter'),
                       Item(one | 63 << WAIT, i, 'sm80: increment by one'),
                       Item(atom | 63 << WAIT, i, 'sm80: shared atomic increment')]
                new[-1].then = SEQUENCE
            elif isinstance(operation, tuple):
                kind, global_offset, shared_offset = self.ampere_ops[i]
                assert kind == 'copy128'
                if self.matrix_at is None:
                    raise Refused('no dedicated register group for an sm_80 asynchronous copy')
                temp = (self.matrix_at + 3) & ~3
                guard = {'Pg': d.ops['Pg'], 'Pg_not': d.ops['Pg@not']}
                load = self.mc.assemble('ldg__sImmOffset', e='E', sz='_128', sem='WEAK', sco='SYS',
                                        cop='EN', Pu=7, Rd=temp, Ra=d.ops['Ra'], Ra_offset=global_offset,
                                        dst_wr_sb=ARRIVES, **guard)
                form = 'sts__uImmOffset' if d.ops['Rb'] == 255 else 'sts__sImmOffset'
                store = self.mc.assemble(form, sz='_128', Ra=d.ops['Rb'], Ra_offset=shared_offset,
                                         Rb=temp, src_rel_sb=SEQUENCE, **guard)
                a = Item(load | (63 << WAIT), i, 'sm80: load asynchronous copy')
                a.then = ARRIVES
                b = Item(store | (63 << WAIT), i, 'sm80: store asynchronous copy')
                b.then = SEQUENCE
                new = [a, b]
            elif self.ampere_ops.get(i) == 'bf16add':
                new = self.bf16_add(i, d)
            elif self.ampere_ops.get(i) == 'bf16pack':
                new = self.bf16_pack(i, d)
            elif self.ampere_ops.get(i) == 'pack':
                # Ra of the surrogate feeds the low half, Rc the high half.
                rd, low, high = (d.ops[k] for k in ('Rd', 'Ra', 'Rc'))
                upper = self.pool(1) if high != 255 else 255
                new = []
                def convert(dest, source, description):
                    word = self.mc.assemble('f2f_f32_downconvert__RRR_RRR', Rd=dest, Rb=source,
                                            **{'dstfmt.srcfmt': 'F16.F32'})
                    item = Item((word & ~(7 << WRITE_BARRIER)) | (ARRIVES << WRITE_BARRIER) | (63 << WAIT), i, description)
                    item.then = ARRIVES
                    new.append(item)
                if high != 255:
                    convert(upper, high, 'sm80: convert high half')
                convert(rd, low, 'sm80: convert low half')
                permute = self.mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=rd, Sb=0x5410, Rc=upper)
                new.append(Item(permute, i, 'sm80: zero high half' if high == 255 else 'sm80: pack both halves'))
            else:
                new = self.one(i, d)
            if self.ampere:
                for item in new:
                    item.changed = item.anew = True
            if i in self._lane_taken:
                # Level 3: the lane is read where a sequence goes by it, and what follows waits.
                lane = Item(mc.assemble('s2r_', Rd=self._lane_taken[i], SRa='SR_LANEID', dst_wr_sb=SEQUENCE), i, 'lane')
                lane.then = SEQUENCE
                new = [lane] + new
            # A frame-backed predicate uses a temporary only while named by this instruction.
            # Its conversion must stay between the save and restore of any borrowed register.
            after = []
            held_in_register = []
            for q, p, reads, writes in self.held.get(i, ()):
                if self.borrow is None or q not in self.borrow.predicate_slots:
                    held_in_register.append((q, p, reads, writes))
                    continue
                word = self.borrow.predicate_slots[q]
                temp = self.take(1)
                if reads:
                    self.load_word(word, temp, i, before, 'mem: load UP%d' % q)
                    before[-1].then = ARRIVES
                    before.append(self.read_back(temp, p, i))
                if writes:
                    after.append(self.written_down(temp, p, i))
                    self.store_word(word, temp, i, after, 'mem: store UP%d' % q)
            new = before + new + after + self.to_memory(i, d)
            if i in self.spilled:
                # Room made (`make_room`): the registers taken from the program are put in the
                # frame before everything the instruction became, and brought back after it.
                # A store reads its operands late: it says when, and what follows waits; what
                # is brought back arrives on a scoreboard, and what follows the last waits.
                b, stl, saves, back = self.borrow, mc.md70.by_name['stl__sImmOffset'], [], []
                for k, r in enumerate(self.spilled[i]):
                    offset = b.depth[i] + 4 * (b.slot_count() + k)
                    saves.append(self.late(stl, mc.assemble('stl__sImmOffset', Ra=1, Rb=r, Ra_offset=offset), i, 'room: R%d away' % r))
                    back.append(Item(mc.assemble('ldl__sImmOffset', Rd=r, Ra=1, Ra_offset=offset, dst_wr_sb=ARRIVES), i, 'room: R%d back' % r))
                back[-1].then = ARRIVES
                b.spill = max(b.spill, len(self.spilled[i]))
                new = saves + new + back
            rel = relative(mc, 75, d)
            if rel is not None:
                target = 16 * i + 16 + signed(rel[0], d.word) * rel[1]
                if target % 16 or not 0 <= target <= 16 * len(self.src):
                    raise Refused('a branch at 0x%x to 0x%x, outside the function' % (16 * i, target))
                # The instruction that keeps the relative operand is the last of the sequence.
                new[-1].target = target
                targets.add(target)
            # The predicates saved for a life that starts here, and given back after one that ends here.
            # The uniform predicates held in a register: read back for the instruction, written after it.
            new = ([self.read_back(self.holder(q), p, i) for q, p, reads, writes in held_in_register if reads] + new
                   + [self.written_down(self.holder(q), p, i) for q, p, reads, writes in held_in_register if writes])
            new = [self.saved(k, p, i) for k, p in self.saves.get(i, ())] + new + [self.given_back(k, p, i) for k, p in self.restores.get(i, ())]
            if self.q is not None:
                # The accumulators of pairs brought to the state the instruction, or what follows it, takes them in.
                new = self.converted(i, self.q.before) + new + self.converted(i, self.q.after)
                # The companions of address registers, made where the plan has them.
                new = self.accompanied(i, self.q.made_before) + new + self.accompanied(i, self.q.made_after)
            if i == 0 and (frame or (self.borrow is not None and self.borrow.room and self._entry_sets_r1)):
                # Level 3: the stack pointer starts under the frame (checked above: this
                # instruction sets it to the top, and nothing else reads the top). (With room
                # to be made, how large the frame is is known at the end: `_lower`.)
                self._lower = Item(mc.assemble('iadd3_imm__RsIR_RIR', Rd=1, Ra=1, Sb=-max(frame, 16), Rc=255), 0, 'mem: R1 -= %d' % frame)
                new = new + [self._lower]
            if i == 0 and self.needs_lane and self.lane_at is not None:
                # Before the first instruction, and part of it: the number of the thread in its
                # warp, which does not change. What follows waits for it. (It is where offset 0
                # now is: a return within the section counts from there, and a branch to the
                # first instruction reads the lane again, which costs little. As an instruction
                # of its own before offset 0, every return of the function went one instruction
                # too far: the test kernels have no call, PyTorch's attention has.)
                first = Item(mc.assemble('s2r_', Rd=self.lane(), SRa='SR_LANEID', dst_wr_sb=SEQUENCE), 0, 'lane')
                first.then = SEQUENCE
                new = [first] + (self.kept() if self.has_product else []) + (self.kept_for_rows() if self.q is not None else []) + new
            if i in self.delayed:
                # Its translation writes or saves a predicate that a late instruction of the
                # program writes (`stand_ins`): first an instruction that does nothing, waits
                # for every scoreboard, and lasts until whatever was issued before has written.
                # (A stall count holds 15.)
                cycles = self.delayed[i]
                nothing = [Item(with_stall(NOP, min(cycles, 15)) | 63 << WAIT, i, 'late predicate')]
                while cycles > 15:
                    cycles -= 15
                    nothing.append(Item(with_stall(NOP, min(cycles, 15)), i, 'late predicate'))
                new = nothing + new
            # What the instruction waited for, the first instruction of what it became waits for:
            # a wait is for everything after it, and the compiler may have put on this one the
            # wait for what a later one reads. (The constants of ULDC lost theirs until
            # 2026-11-17: the test `translated-waits-moved` of tests/gpu/run.sh is for that.)
            new[0].word |= d.word & (63 << WAIT)
            # Borrowed registers: what the program's instructions wrote there last may still be on
            # its way. The first instruction of the translation that writes a borrowed register
            # waits for all six scoreboards.
            # (What the instruction was given, `take`: the pool, and at level 3 the rest; and
            # at level 3 the register that holds a uniform predicate, where it is written.)
            at_instr = {r for r in range(2, 255) if self.given >> r & 1}
            if self.borrow is not None:
                at_instr.update(self.hold_at[q] for q, p, reads, writes in self.held.get(i, ()) if writes and self.hold_at.get(q, 255) <= self.borrow.top)
            if self.borrow is not None and (i in self.pool_taken or at_instr or any(l == 'U' and u in self.borrow.homes for l, u in d.writes)):
                # It writes into a register of the program that holds nothing: what the program
                # wrote there last may still be on its way (the words of a load that nothing
                # reads), and would be written over what is put there now. Every
                # scoreboard is waited for first. (Two cycles after whatever names one: `schedule`.)
                new[0].word |= 63 << WAIT
                mine = {self.borrow.homes[u] for l, u in d.writes if l == 'U' and u in self.borrow.homes}
                if i in self.pool_taken:
                    mine.update((self.pool_taken[i], self.pool_taken[i] + 1))
                mine.update(at_instr)
                for item in new:
                    item.borrows = mine
            new[0].first = True
            items += new
        if self.borrow is not None and self.borrow.room:
            # The frame with the room made: the uniform registers, then as many words as an
            # instruction put away at once.
            b = self.borrow
            b.frame_bytes = (4 * (b.slot_count() + b.spill) + 15) & ~15
            if b.frame_bytes and self._top_again is not None:
                raise Refused('a frame for the translation, and the top of the stack is read again at 0x%x' % (16 * self._top_again))
            if self._lower is not None:
                if b.frame_bytes:
                    self._lower.word = mc.assemble('iadd3_imm__RsIR_RIR', Rd=1, Ra=1, Sb=-b.frame_bytes, Rc=255) | (self._lower.word & (63 << WAIT))
                    self._lower.what = 'mem: R1 -= %d' % b.frame_bytes
                else:
                    items.remove(self._lower)
        for branch, places in self.tables.items():
            if branch % 16 or branch >= 16 * len(self.src) or self.src[branch // 16].c.name != 'brx_':
                raise Refused('a jump table for 0x%x, which is not a branch through a register' % branch)
            if any(t % 16 or not 0 <= t < 16 * len(self.src) for t in places):
                raise Refused('the jump table of the branch at 0x%x leads outside the function' % branch)
            targets.update(places)
        if self.borrow is not None and self.borrow.used:
            used = self.borrow.used
            for item in items:
                d70 = mc.decode(70, item.word)
                if d70 is None:
                    continue
                board = (item.word >> WRITE_BARRIER) & 7
                written = {r for l, r in d70.writes if l == 'R'}
                if written & used and unannounced(d70) and not item.covered:
                    # (`Borrowed` leaves out the registers the program writes so; this is for
                    # what the translation made of an instruction. A covered instruction is
                    # part of a sequence whose later instruction says when the result is there,
                    # and what follows waits for it: the late arrival is handled.)
                    raise Refused('a borrowed register receives at 0x%x a result that nothing announces' % (16 * item.origin))
                if board != 7 and written.intersection(item.borrows):
                    # A result of the translation that is written when it arrives, in a register
                    # the program may write next (where nothing reads it on some way, it would
                    # arrive over what the program wrote): the next instruction waits until it
                    # is there.
                    item.also.append(board)
                # An instruction that reads a borrowed register some time after it is issued,
                # for the program: it says on a scoreboard when it has read, so that what the
                # translation writes there next, which waits for every scoreboard, comes after.
                # (The compiler made it say so only where the program itself writes the register
                # again soon. Those that read for the translation are `late`: they say so and
                # the next instruction waits, since the program's next write waits for nothing.)
                if not item.late and d70.kind in MC.LATE and (item.word >> READ_BARRIER) & 7 == 7 and any(l == 'R' and r in used for l, r, _ in d70.reads):
                    item.word = self.read_announced(d70.c, self.unmarked(item.word, item.what), item.origin)
                    item.says = True
        self.items, self.targets, self.relocate = items, targets, []
        for k, item in enumerate(items):
            if item.late:
                # (`late`.) The instruction after it waits until it has read.
                item.then = (item.word >> READ_BARRIER) & 7
            if item.then is not None or item.also:
                if k + 1 == len(items):
                    raise Refused('a function that ends on a memory instruction')
                for board in ([item.then] if item.then is not None else []) + item.also:
                    items[k + 1].word |= 1 << (WAIT + board)
        for item in items:
            item.d = mc.decode(70, item.word)
            if item.d is None:
                raise Refused('the translation of the instruction at 0x%x (%s) is not an sm_70 instruction: %032x' % (16 * item.origin, self.src[item.origin].c.name, item.word))
        self.return_addresses()
        self.schedule()
        self.check_lepc_pairs()
        return self.finish()

    def liveness(self):
        """Which predicates hold a value that may still be read, before and after each instruction:
        bits 0 to 6 the ordinary ones, bits 7 to 13 the uniform ones. A return may be followed by
        anything (everything is alive there), an exit by nothing. A return within the section
        (RET.REL) goes back to after one of the section's own calls."""
        mc, n = self.mc, len(self.src)
        returns_to = [i + 1 for i, d in enumerate(self.src) if d.c.name.startswith('call_rel') and i + 1 < n]
        self.defs, self.uses, self.external = [0] * n, [0] * n, [False] * n
        kills, successors, ends = [0] * n, [[] for _ in range(n)], [0] * n
        everything = (1 << 14) - 1
        for i, d in enumerate(self.src):
            ops = d.ops
            true = lambda p: ops.get(p, 7) == 7 and not ops.get(p + '@not', 0)
            unguarded = true('Pg') and true('UPg')
            for letter, register, role in d.reads:
                if letter in 'PQ':
                    self.uses[i] |= 1 << (register + (7 if letter == 'Q' else 0))
            for letter, register in d.writes:
                if letter in 'PQ':
                    self.defs[i] |= 1 << (register + (7 if letter == 'Q' else 0))
            name = d.c.name
            # The predicates that P2R reads and R2P writes are in a mask: a number, or else any.
            # (What R2P writes is not taken to end the life of what the predicate held: nvdisasm,
            # which validate.py checks the predicates against, does not take it so either.)
            mask = ops['Sb'] & 0x7f if name in ('p2r__RsIR_RIR', 'r2p__RIR') else 0x7f
            if d.mnemonic == 'P2R':
                self.uses[i] |= mask
            if d.mnemonic == 'R2P':
                self.defs[i] |= mask
            elif unguarded:
                kills[i] = self.defs[i]
            rel = relative(mc, 75, d)
            target = 16 * i + 16 + signed(rel[0], d.word) * rel[1] if rel is not None else None
            falls = True
            if d.mnemonic == 'EXIT' and unguarded:
                falls = False
            elif name == 'ret__REL':
                falls, target = not unguarded, None
                successors[i] += returns_to
            elif name == 'brx_' and 16 * i in self.tables:
                # To one of the places its table lists.
                falls, target = not unguarded, None
                successors[i] += [t // 16 for t in self.tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n]
            elif d.mnemonic == 'RET' or d.mnemonic in ('BRX', 'JMX', 'JMP'):
                falls, ends[i] = not unguarded, everything
                target = None
            elif d.mnemonic == 'BRA' and unguarded and true('Pp') and not ops.get('cond'):
                # (BRA.DIV and BRA.CONV go by whether the threads of the warp are together: the
                # code after them is where the others go. Taken as always taken until 2026-11-17:
                # the helper that a warp calls when it is not together was left out of the lives.)
                falls = False
            elif name.startswith('call_abs'):
                self.external[i] = True
            if falls and i + 1 < n:
                successors[i].append(i + 1)
            if target is not None and target % 16 == 0 and 0 <= target < 16 * n:
                successors[i].append(target // 16)
        self.successors, self.kills = successors, kills
        self.live_in, self.live_out = [0] * n, list(ends)
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                out = ends[i]
                for s in successors[i]:
                    out |= self.live_in[s]
                into = self.uses[i] | (out & ~kills[i])
                if out != self.live_out[i] or into != self.live_in[i]:
                    self.live_out[i], self.live_in[i], changed = out, into, True

    def saved(self, k, p, i):
        """The item that saves predicate `p` in the `k`th register kept for that."""
        if self.save_at is None:
            raise Refused('a predicate to save in a function that was not given a register for it')
        item = Item(self.mc.assemble('p2r__RsIR_RIR', Rd=self.save_at + k, Ra=255, Sb=1 << p), i, 'save')
        item.more_reads = [('P', p, 'data')]
        return item

    def given_back(self, k, p, i):
        """The item that gives predicate `p` back from the `k`th register kept for that."""
        if self.save_at is None:
            raise Refused('a predicate to save in a function that was not given a register for it')
        item = Item(self.mc.assemble('r2p__RIR', Ra=self.save_at + k, Sb=1 << p), i, 'restore')
        item.more_writes = [('P', p)]
        return item

    def holder(self, q):
        """The register that holds uniform predicate `q` between the instructions that name it, in
        the lives where no ordinary predicate stands for it all along (`stand_ins`)."""
        if self.hold_at is None or q not in self.hold_at:
            raise Refused('a uniform predicate to hold in a function that was not given a register for it')
        return self.hold_at[q]

    def read_back(self, holder, p, i):
        """Put the held value in predicate `p`: whether its register is not zero."""
        return Item(self.mc.assemble('isetp__RRR_RR', Pu=p, Ra=holder, Rb=255, icmp='NE', fmt='U32', bop='AND'), i, 'held: back')

    def written_down(self, holder, p, i):
        """Write predicate `p` to the register that holds it: its bit, alone."""
        item = Item(self.mc.assemble('p2r__RsIR_RIR', Rd=holder, Ra=255, Sb=1 << p), i, 'held: kept')
        item.more_reads = [('P', p, 'data')]
        return item

    def bf16_round(self, i, new, what, to, source, other):
        """Appends to `new` the instructions that leave in `to` the bits of the bf16 nearest to
        the f32 in `source` (ties to even), 0x7fff for a NaN: the upper half of the f32, rounded
        on its lower half. `to` may be `source`; `other` is written."""
        mc = self.mc
        def add(word, step):
            new.append(Item(word, i, 'sm80: %s: %s' % (what, step)))
        def right(to, source, by, fmt='U32'):
            return mc.assemble('shf__RuIR_RIR', Rd=to, Ra=255, Sb=by, Rc=source, dir='R', fmt=fmt, hilo='HI')
        def logic(to, a, immediate, c, lut):
            return mc.assemble('lop3_lut__RuIR_RIR', Rd=to, Ra=a, Sb=immediate, Rc=c, imm8=lut)
        add(right(other, source, 16), 'the bits that stay')
        add(logic(other, other, 1, 255, 0xc0), 'the lowest of them')
        add(mc.assemble('iadd3_imm__RsIR_RIR', Rd=other, Ra=source, Sb=0x7fff, Rc=other), 'to nearest, ties to even')
        add(right(other, other, 16), 'rounded')
        add(logic(to, source, 0x7fffffff, 255, 0xc0), 'magnitude')
        add(mc.assemble('iadd3_imm__RsIR_RIR', Rd=to, Ra=to, Sb=-0x7f800001, Rc=255), 'negative unless a NaN')
        add(right(to, to, 31, 'S32'), 'ones unless a NaN')
        add(logic(to, other, 0x7fff, to, 0xe4), '0x7fff for a NaN')

    def bf16_add(self, i, d):
        """sm_80's HFMA2.BF16_V2 Rd, Ra, 1, 1, Rc: in each half of the register, the sum of two
        bf16, rounded to bf16 once. Each half is widened to an f32 (a bf16 is the upper half of
        one), the two are added as f32, and the sum is rounded to bf16 (`bf16_round`). Rounding
        the f32 sum gives what one rounding of the exact sum gives (docs/emulations.md, "bf16")."""
        if self.matrix_at is None:
            raise Refused('no dedicated registers for an sm_80 bf16 sum')
        mc, (s0, s1, s2) = self.mc, range(self.matrix_at, self.matrix_at + 3)
        rd, ra, rc = (d.ops[k] for k in ('Rd', 'Ra', 'Rc'))
        new = []
        def add(word, what):
            new.append(Item(word, i, 'sm80: bf16 sum: ' + what))
        def left(to, source):
            return mc.assemble('shf__RuIR_RIR', Rd=to, Ra=source, Sb=16, Rc=255, dir='L', fmt='U32')
        def upper(to, source):
            return mc.assemble('lop3_lut__RuIR_RIR', Rd=to, Ra=source, Sb=0xffff0000, Rc=255, imm8=0xc0)
        add(left(s0, ra), 'low half of a as f32')
        add(left(s1, rc), 'low half of c as f32')
        add(mc.assemble('fadd__RRR_RR', Rd=s0, Ra=s0, Rc=s1), 'low halves added')
        add(upper(s1, ra), 'high half of a as f32')
        add(upper(s2, rc), 'high half of c as f32')
        add(mc.assemble('fadd__RRR_RR', Rd=s1, Ra=s1, Rc=s2), 'high halves added')
        self.bf16_round(i, new, 'bf16 sum, low', s0, s0, s2)
        self.bf16_round(i, new, 'bf16 sum, high', s1, s1, s2)
        add(mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=s0, Sb=0x5410, Rc=s1), 'both halves')
        return new

    def f2fp_f16(self, i, d, guard):
        """Turing's F2FP.F16.F32 packs two f32 values into two f16 halves of one register.
        Volta's F2F.F16.F32 converts one; two of them and a PRMT pack the pair. Only the plain
        rounding to f16 is translated."""
        ops = d.ops
        if ops.get('dstfmt', 0) or ops.get('satfinite', 0) or ops.get('relu', 0) or ops.get('merge', 0) or ops.get('rndMode', 0):
            raise Refused('F2FP with a destination format or a rounding that is not translated at 0x%x' % (16 * i))
        rd, high, low = ops['Rd'], ops['Ra'], ops['Rb']
        if rd == 255 or low == 255:
            raise Refused('F2FP without a destination or a low source at 0x%x' % (16 * i))
        new = []
        if high == 255:
            upper = 255
        else:
            try:
                upper = self.pool(1)
            except Refused:
                upper = self.next_register
                self.next_register += 1
        def convert(dest, source, guarded, what):
            word = self.mc.assemble('f2f_f32_downconvert__RRR_RRR', Rd=dest, Rb=source,
                                    **{'dstfmt.srcfmt': 'F16.F32'}, **(guard if guarded else {}))
            item = Item((word & ~(7 << WRITE_BARRIER)) | (ARRIVES << WRITE_BARRIER) | (63 << WAIT), i, what)
            item.then = ARRIVES
            new.append(item)
        if high != 255:
            convert(upper, high, False, 'f2fp: convert high half')
        convert(rd, low, True, 'f2fp: convert low half')
        permute = self.mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=rd, Sb=0x5410, Rc=upper, **guard)
        permute = (permute & ~(7 << WRITE_BARRIER)) | ((d.word >> WRITE_BARRIER) & 7) << WRITE_BARRIER
        new.append(Item(permute, i, 'f2fp: pack halves'))
        return new

    def bf16_pack(self, i, d):
        """sm_80's F2FP.BF16.PACK_AB Rd, Ra, Rb: the f32 of Rb rounded to bf16 in the low half,
        that of Ra in the high half. (Ra of the surrogate is the low one, Rc the high one.)"""
        if self.matrix_at is None:
            raise Refused('no dedicated registers for an sm_80 bf16 conversion')
        mc, (s0, s1, s2) = self.mc, range(self.matrix_at, self.matrix_at + 3)
        rd, low, high = (d.ops[k] for k in ('Rd', 'Ra', 'Rc'))
        new = []
        self.bf16_round(i, new, 'bf16 pack, low', s0, low, s2)
        if high != 255:
            self.bf16_round(i, new, 'bf16 pack, high', s1, high, s2)
        new.append(Item(mc.assemble('prmt__RuIR_RIR', Rd=rd, Ra=s0, Sb=0x5410, Rc=s1 if high != 255 else 255), i, 'sm80: bf16 pack: both halves'))
        return new

    def pool(self, width=2):
        """The first of the two registers that the sequences of the translation compute into. Each
        sequence writes what it needs of them and reads it before it ends, and an instruction that
        reads them late is waited for (`late`): one pair serves every sequence. `width` 1: the
        sequence writes the first one only."""
        if self.pool_at is None and self.borrow is not None:
            # (Borrowed for this instruction: `take`. What the sequence writes first waits for
            # everything, `translate`.)
            if self.at not in self.pool_taken:
                self.pool_taken[self.at] = self.take(width, 'no %s nothing at 0x%x' % ('register holds' if width == 1 else 'two registers side by side hold', 16 * self.at))
                self.pool_width[self.at] = width
            elif width > self.pool_width[self.at]:
                raise Refused('the sequence at 0x%x takes one register to compute into, then two' % (16 * self.at))
            return self.pool_taken[self.at]
        if self.pool_at is None:
            raise Refused('a sequence that needs registers in a function that was not given any')
        return self.pool_at

    def take(self, width, why=None):
        """Registers for the translation of the instruction at hand, among those of the program
        that hold nothing there: the lowest (`Borrowed.free_at`) that this instruction was not
        given already. The pool of level 2, and at level 3 everything."""
        r, made = self.borrow.free_at(self.at, width, self.given), False
        if r is None and self.borrow.room:
            r, made = self.make_room(width), True
        if r is None:
            raise Refused(why or 'no %s nothing at 0x%x' % ('register holds' if width == 1 else '%d registers in a row hold' % width, 16 * self.at))
        self.given |= ((1 << width) - 1) << r
        if made:
            self.spilled.setdefault(self.at, []).extend(range(r, r + width))
        else:
            self.given_at.setdefault(self.at, []).append((r, width))
        self.borrow.used.update(range(r, r + width))
        return r

    def make_room(self, width):
        """Where no register holds nothing at the instruction at hand (level 3, `VF_BORROW=5`):
        registers that the instruction does not touch (`Borrowed.victims_at`), which
        `translate` puts in the frame before the sequence and brings back after it. Only at an
        instruction that always goes on to the next one (what is put away must come back),
        after the first (which sets the stack pointer the frame is addressed from). None when
        it cannot be."""
        i = self.at
        d = self.src[i]
        if i == 0 or not self._entry_sets_r1 or self.borrow.depth[i] is None or any(l == 'R' and r == 1 for l, r in d.writes):
            # (Nor at an instruction that moves the stack pointer: what is put away before it
            # would be looked for elsewhere after it.)
            return None
        if not (self.successors[i] == [i + 1] and not self.external[i] and (d.kind != 'BRU' or d.mnemonic in ('BSSY', 'BMOV'))):
            return None
        return self.borrow.victims_at(i, width, self.given)

    def lane(self):
        """The register that holds the lane of the thread (0 to 31): all through the function,
        or at level 3 one taken at the instruction, which `translate` has read there."""
        if self.lane_at is not None:
            return self.lane_at
        if self.borrow is not None and self.borrow.everything:
            if self.at not in self._lane_taken:
                self._lane_taken[self.at] = self.take(1)
            return self._lane_taken[self.at]
        raise Refused('a sequence that goes by the lane in a function that was not given a register for it')

    def scratch(self):
        """The first of the registers that the sequence of a matrix instruction computes into
        (`MATRIX_SCRATCH` of them). Like the pool, they hold nothing from one sequence to the
        next: a sequence ends when every instruction of it that reads late has read
        (`matrix_load`). At level 3: that many in a row from an even one, at the instruction."""
        if self.matrix_at is not None:
            return self.matrix_at
        if self.borrow is not None and self.borrow.everything:
            if self.at not in self._scratch_taken:
                self._scratch_taken[self.at] = self.take(MATRIX_SCRATCH)
            return self._scratch_taken[self.at]
        raise Refused('a sequence that needs registers of its own in a function that was not given them')

    def frame_top(self, i, items):
        """The register that holds, for the instruction at hand, the top of the thread's local
        memory (c[0x0][0x28]; the frame of the translation is right under it): loaded once, by
        an item added to `items`. Not the stack pointer: the program moves that one."""
        if self._base is None:
            self._base = self.take(1)
            items.append(Item(self.mc.assemble('mov__RCR_RC', Rd=self._base, Sb_bank=0, Sb_addr=0x28 // 4), i, 'mem: base'))
        return self._base

    def in_frame(self, d):
        """The uniform registers of the frame that instruction `d` reads, those it writes, and
        the lower one of each two that it takes as one operand of 64 bits."""
        slots, uses = self.borrow.slots, self.mc._uses[(75, d.c.name)]
        read, written, together = set(), set(), set()
        for operands, found in ((uses.reads, read), (uses.writes, written)):
            for letter, operand, span in operands:
                if letter == 'U' and operand in d.ops and d.ops[operand] != 63:
                    u, count = d.ops[operand], MC.registers(span, d.ops)
                    found.update(u + k for k in range(count) if u + k in slots)
                    if count == 2 and u in slots:
                        together.add(u)
        return read, written, together

    def from_memory(self, i, d):
        """Level 3, before the instruction at `i`: a register, taken there, for each uniform
        register of the frame that the instruction reads or writes (`R` gives it), two side by
        side from an even one where the instruction takes a pair as one operand; and what it
        reads loaded from the frame. An instruction under a guard may leave what it writes as it
        was: that is loaded too, and what is stored after it is then right either way. Every
        load says on a scoreboard when its result is there, and what follows the last one waits."""
        if self.borrow is None or not self.borrow.slots:
            return []
        read, written, together = self.in_frame(d)
        if not pairing.unguarded(d):
            read = read | written
        if not (read or written):
            return []
        items = []
        for u in sorted(read | written):
            if u in self._mem_regs:
                continue
            if u in together:
                first = self.take(2)
                self._mem_regs[u], self._mem_regs[u + 1] = first, first + 1
            else:
                self._mem_regs[u] = self.take(1)
        if read:
            for u in sorted(read):
                self.load_word(self.borrow.slots[u], self._mem_regs[u], i, items, 'mem: load UR%d' % u)
            items[-1].then = ARRIVES
        return items

    def slot(self, word, i, items):
        """Where word `word` is in the frame, for the instruction at hand: the register
        to address it from and the offset. From the top of the thread's local memory, loaded in
        a register (`frame_top`); or, where it is known how far the program has moved its stack
        pointer at each instruction and room is made (`Borrowed.from_r1`, `depth`), from the
        stack pointer itself, which the translation lowered by the frame: no register, no load."""
        if self.borrow.from_r1:
            if self.borrow.depth[i] is None:
                raise Refused('a uniform register in memory at 0x%x, where the stack pointer is not known' % (16 * i))
            return 1, self.borrow.depth[i] + 4 * word
        return self.frame_top(i, items), self.borrow.slot_offset(word) & 0xFFFFFF

    def load_word(self, word, rd, i, items, what):
        top, offset = self.slot(word, i, items)
        items.append(Item(self.mc.assemble('ldl__sImmOffset', Rd=rd, Ra=top, Ra_offset=offset, dst_wr_sb=ARRIVES), i, what))

    def store_word(self, word, rb, i, items, what):
        top, offset = self.slot(word, i, items)
        stl = self.mc.md70.by_name['stl__sImmOffset']
        items.append(self.late(stl, self.mc.assemble('stl__sImmOffset', Ra=top, Rb=rb, Ra_offset=offset), i, what))

    def to_memory(self, i, d):
        """Level 3, after the instruction at `i`: the uniform registers of the frame that it
        wrote are stored there, under no guard. An instruction may write the predicate that
        guards it (`@UP0 UIADD3 UR0, UP0, URZ, 0x0, URZ`: its carry), and a store under that
        guard would go by what the instruction left in it. Where the instruction is under a
        guard, what it writes was loaded before it (`from_memory`): the store puts back what
        was there when the instruction did not run. A store reads its operands some time after
        it is issued: it says when, and what follows waits (`late`)."""
        if self.borrow is None or not self.borrow.slots:
            return []
        read, written, together = self.in_frame(d)
        if not written:
            return []
        items = []
        for u in sorted(written):
            self.store_word(self.borrow.slots[u], self._mem_regs[u], i, items, 'mem: store UR%d' % u)
        return items

    def entries(self):
        """The instructions that are not only entered from the one before them: where the
        relative operand of an instruction leads (a branch, a call within the section, the place
        where the threads of a warp are to meet again), where the table of a branch through a
        register leads, and what follows a call within the section when the section has a return
        for it. None when that cannot be known: a branch through a register that has no table,
        a jump to an address."""
        mc, n, out = self.mc, len(self.src), set()
        returns = any(d.c.name == 'ret__REL' for d in self.src)
        for i, d in enumerate(self.src):
            if d.c.name.startswith(('brx', 'jmx')):
                # Through a register: where its table says, or anywhere.
                if d.c.name != 'brx_' or 16 * i not in self.tables:
                    return None
                out.update(t // 16 for t in self.tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n)
                continue
            rel = relative(mc, 75, d)
            if rel is not None:
                target = 16 * i + 16 + signed(rel[0], d.word) * rel[1]
                if target % 16 == 0 and 0 <= target < 16 * n:
                    out.add(target // 16)
            elif d.c.name.startswith('jmp'):
                return None
            if returns and d.c.name.startswith('call_rel') and i + 1 < n:
                out.add(i + 1)
        return out

    def kept(self):
        """What a function with a product computes of the lane when it is entered, into four
        registers that nothing else writes: the two lanes that hold what a thread gathers of B
        ((lane >> 4) * 8 + thread, and 16 more), all ones in the upper half of each sixteen
        lanes (bit 3 of the lane), and all ones in the threads with an odd number. (Each product
        computed the four again until 2026-11-18: eight instructions of its thirty-three.)"""
        if self.kept_at is None:
            raise Refused('a product in a function that was not given registers for what it keeps of the lane')
        asm, lane = self.mc.assemble, self.lane()
        low, high, half, odd = range(self.kept_at, self.kept_at + KEPT)
        shf = lambda **k: asm('shf__RuIR_RIR', **{'fmt': 'U32', **k})
        lut = lambda **k: asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, **k)
        words = [shf(Rd=low, Ra=255, Sb=1, Rc=lane, dir='R', hilo='HI'),
                 lut(Rd=low, Ra=low, Sb=8, Rc=255, imm8=0xc0),
                 lut(Rd=low, Ra=low, Sb=3, Rc=lane, imm8=0xf8),
                 lut(Rd=high, Ra=low, Sb=16, Rc=255, imm8=0xfc),
                 shf(Rd=half, Ra=lane, Sb=28, Rc=255, dir='L', hilo='LO'),
                 shf(Rd=half, Ra=255, Sb=31, Rc=half, dir='R', hilo='HI', fmt='S32'),
                 shf(Rd=odd, Ra=lane, Sb=31, Rc=255, dir='L', hilo='LO'),
                 shf(Rd=odd, Ra=255, Sb=31, Rc=odd, dir='R', hilo='HI', fmt='S32')]
        return [Item(word, 0, 'kept') for word in words]

    def kept_gathering(self, i, rb):
        """Whether the product at `i`, on the register `rb` of B, finds in the registers kept
        for that what it would gather: the last product that gathered did it from the same
        register; every way to this one goes through that one and straight on from it (nothing
        after it up to this one is entered otherwise than from the instruction before:
        `entries`); no call between them (the function called may gather for a product of its
        own, or write the register); and nothing from that product on, itself included, writes
        the register, under a guard or not.
        At level 3 (everything at the instruction): always no. Nothing is kept between
        instructions; every product gathers for itself."""
        if self.borrow is not None and self.borrow.everything:
            return False
        if self.holds is None or self.entered is None:
            return False
        gathered, held = self.holds
        if held != rb or any(j in self.entered for j in range(gathered + 1, i + 1)):
            return False
        return not any(d.c.name.startswith('call') or ('R', rb) in d.writes for d in self.src[gathered:i])

    def late(self, c, word, i, what):
        """The item for `word`, of sm_70 class `c`: an instruction that reads its operands some time
        after it is issued (a memory access, a conversion, arithmetic on doubles: `machine.LATE`),
        and that the translation gave a register to read: the stand-in of a uniform register, or
        the registers of the pool.

        Turing has a uniform value when the instruction is issued, and the program may change the
        uniform register right after it; and the pool is written again by the next sequence that
        needs it. Native code keeps what such an instruction reads until it has read: the
        instruction says so on a scoreboard, and what writes one of its operands next waits for
        that (rdbar2.py: no write of an operand of a pending load without such a wait in native
        sm_70 code). Here the instruction says so, and the very next instruction waits for it: by
        then nothing it reads can change, whatever comes after."""
        item = Item(self.read_announced(c, self.unmarked(word, what), i), i, what)
        item.late = True
        return item

    def unmarked(self, word, what):
        """`word`, an sm_70 instruction of kind `what`, without its marks of register reuse. In
        an instruction of the program (kept as it is, or on ordinary registers) the three bits
        are marks only when its operands can bear them and its stall ends a group
        (`machine.with_stall`); otherwise they name a batch, which is no hint and stays. (Cleared
        whatever they were until 2026-11-20: a texture load of the program that reads a borrowed
        register lost its batch.) What the translation adds names no batch."""
        if what in ('same', 'ordinary registers') and not (marked(self.mc.decode(70, word)) and 17 <= (word >> USCHED) & 31 <= 27):
            return word
        return word & ~REUSE

    def one(self, i, d):
        """The sm_70 instructions for source instruction `i`."""
        mc, name, w = self.mc, d.c.name, d.word
        if mc.decode(70, w) is not None and not any(l in 'UQ' for l, *_ in d.reads + [x + (None,) for x in d.writes]):
            return [Item(w, i, 'same', changed=False)]
        ops = d.ops
        guard = {'Pg': ops['Pg'], 'Pg_not': ops.get('Pg@not', 0)} if 'Pg' in ops else {}
        wait = w & (63 << WAIT)
        if name == 'f2fp__RRR':
            return self.f2fp_f16(i, d, guard)
        if name in ('ldsm__sImmOffset', 'ldsm__URsIR'):
            if self.q is not None and i in self.q.loads:
                return self.whole_rows(i, d, guard, wait)
            return self.matrix_load(i, d, guard, wait)
        if name == 'hmma_x8_':
            if self.q is not None and i in self.q.sites:
                return self.half_of_pair(i, d, wait)
            return self.matrix_product(i, d, wait)
        if name.startswith('imad_hi__'):
            # The upper half of a 64-bit product-sum: the whole of it in two registers of the
            # translation, then the upper one where the program wants it.
            # Its carry, if it asks for one, is the carry of the same 64-bit sum: the product gives it.
            # (Into the predicate that guards the instruction, the move after the product would
            # go by the carry: no code seen does that.)
            if ops.get('Pu', 7) != 7 and ops['Pu'] == ops.get('Pg', 7):
                raise Refused('IMAD.HI with its carry into the predicate that guards it at 0x%x' % (16 * i))
            wide = self.rule(d, 'imad_wide__' + name[len('imad_hi__'):].replace('RUR_RUR', 'RRR_RRR'), i)
            c70 = mc.md70.by_name[mc.decode(70, wide).c.name]
            wide = self.put(c70, wide, 'Rd', self.pool())
            move = mc.assemble('mov__RRR_RR', Rd=ops['Rd'], Rb=self.pool() + 1, **guard)
            move = (move & ~(31 << USCHED)) | (w & (31 << USCHED))
            return [Item(wide & ~REUSE, i, 'product'), Item(move, i, 'upper half')]
        if name.startswith('imnmx__'):
            # The smaller or the larger of two: a comparison into a predicate nothing else uses, then
            # a selection by it.
            if ops['Pp'] != 7:
                raise Refused('IMNMX under a predicate that is not constant at 0x%x' % (16 * i))
            (p, k), larger = self.scratch_at[i], ops.get('Pp@not', 0)
            fmt = next(n for n, v in mc.md75.enums[d.c.operands['fmt'].type].items() if v == ops['fmt'])
            if name.endswith('__RRR_RRR') or name.endswith('__RUR_RUR'):
                b = self.R(ops['URb']) if 'URb' in ops else ops['Rb']
                compare = mc.assemble('isetp__RRR_RR', Pu=p, Ra=ops['Ra'], Rb=b, icmp='LT', fmt=fmt, bop='AND')
                select = mc.assemble('sel__RRR_RRR', Rd=ops['Rd'], Ra=ops['Ra'], Rb=b, Pp=p, Pp_not=larger, **guard)
            elif name.endswith('__RsIR_RIR'):
                compare = mc.assemble('isetp__RsIR_RI', Pu=p, Ra=ops['Ra'], Sb=ops['Sb'], icmp='LT', fmt=fmt, bop='AND')
                select = mc.assemble('sel__RuIR_RIR', Rd=ops['Rd'], Ra=ops['Ra'], Sb=ops['Sb'], Pp=p, Pp_not=larger, **guard)
            elif name.endswith('__RCR_RCR'):
                bank = {'Sb_bank': ops['Sb_bank'], 'Sb_addr': ops['Sb_addr']}
                compare = mc.assemble('isetp__RCR_RC', Pu=p, Ra=ops['Ra'], icmp='LT', fmt=fmt, bop='AND', **bank)
                select = mc.assemble('sel__RCR_RCR', Rd=ops['Rd'], Ra=ops['Ra'], Pp=p, Pp_not=larger, **bank, **guard)
            else:
                raise Refused('this form of IMNMX: %s' % name)
            select = (select & ~(31 << USCHED)) | (w & (31 << USCHED)) | (w & REUSE if not name.endswith('RUR') else 0)
            new = [Item(compare | wait, i, 'comparison'), Item(select | wait, i, 'selection')]
            # (Every predicate holds something there: the one it computes into is saved around it.)
            return new if k is None else [self.saved(k, p, i)] + new + [self.given_back(k, p, i)]
        if name in STRIDED and ops.get('stride'):
            # An address with a stride: the register times the stride in the pool, then the plain
            # instruction on it.
            scale = mc.assemble('shf__RuIR_RIR', Rd=self.pool(1), Ra=ops['Ra'], Sb=STRIDES[ops['stride']], Rc=255, dir='L', fmt='U32')
            plain = self.put(d.c, self.put(d.c, w, 'stride', 0), 'Ra', self.pool(1))
            return [Item(scale | wait, i, 'stride'), self.late(mc.md70.by_name[name], plain, i, 'on the product')]
        if name == 'uldc_const__RCR':
            # A constant of the program into a uniform register: into the ordinary one, 32 bits at a time.
            sizes = {n.strip('_') for n, v in mc.md75.enums[d.c.operands['sz'].type].items() if v == ops['sz']}
            guard = {'Pg': self.P(ops['UPg']), 'Pg_not': ops.get('UPg@not', 0)}
            small = sizes & {'U8', 'S8', 'U16', 'S16'}
            if small:
                # One or two bytes of a word of the constants: the word, moved up until the bytes are
                # at its top, then down again with zeros or with the sign.
                kind = small.pop()
                bits, at = int(kind[1:]), ops['Sa_addr'] % 4
                if at * 8 + bits > 32:
                    raise Refused('ULDC of bytes that straddle two words at 0x%x' % (16 * i))
                r = self.R(ops['URd'])
                out = [Item(mc.assemble('mov__RCR_RC', Rd=r, Sb_bank=ops['Sa_bank'], Sb_addr=ops['Sa_addr'] // 4, **guard), i, 'constant')]
                up = 32 - bits - 8 * at
                if up:
                    out.append(Item(mc.assemble('shf__RuIR_RIR', Rd=r, Ra=r, Sb=up, Rc=255, dir='L', fmt='U32', **guard), i, 'bytes'))
                out.append(Item(mc.assemble('shf__RuIR_RIR', Rd=r, Ra=255, Sb=32 - bits, Rc=r, dir='R', fmt='S32' if kind[0] == 'S' else 'U32', hilo='HI', **guard), i, 'bytes'))
                out[-1].word = (out[-1].word & ~(31 << USCHED)) | (w & (31 << USCHED))
                return out
            if not sizes & {'32', '64'}:
                raise Refused('ULDC of %s bits at 0x%x' % ('/'.join(sorted(sizes)), 16 * i))
            if ops['Sa_addr'] % 4:
                raise Refused('ULDC of a constant that is not at a multiple of 4 at 0x%x' % (16 * i))
            out = []
            for half in range(2 if '64' in sizes else 1):
                # ULDC gives the address in bytes, MOV in words of four.
                move = mc.assemble('mov__RCR_RC', Rd=self.R(ops['URd'] + half), Sb_bank=ops['Sa_bank'], Sb_addr=ops['Sa_addr'] // 4 + half, **guard)
                out.append(Item(move, i, 'constant'))
            out[-1].word = (out[-1].word & ~(31 << USCHED)) | (w & (31 << USCHED))
            return out
        if name == 'uldc_ur_offset_':
            # A constant at a place that a uniform register gives. Turing has it at once. On Volta a
            # load of constants brings it some cycles later and says so on a scoreboard, which
            # what follows was not written to wait for: an instruction that does nothing waits for
            # it there. (By then the load has long read the register: it needs no copy.)
            if ops['URa'] == 63:
                raise Refused('ULDC at an index that is the zero register at 0x%x' % (16 * i))
            load = mc.assemble('ldc__RaNonRZ', Pg=self.P(ops['UPg']), Pg_not=ops.get('UPg@not', 0), sz=ops['sz'], Rd=self.R(ops['URd']),
                               Ra=self.R(ops['URa']), Sa_bank=ops['Sa_bank'], Ra_offset=ops['Sa_offset'], dst_wr_sb=ARRIVES)
            arrived = (with_stall(NOP, 1) & ~(31 << USCHED)) | (w & (31 << USCHED)) | (1 << (WAIT + ARRIVES))
            return [Item(load | wait, i, 'constant at an index'), Item(arrived, i, 'until it is there')]
        for prefix, plain in UNIFORM_ADDRESS.items():
            if name.startswith(prefix):
                return self.uniform_address(i, d, plain)
        if name == 'r2ur__noOR' and ops.get('Pu', 7) == 7:
            move = mc.assemble('mov__RRR_RR', Rd=self.R(ops['URd']), Rb=ops['Ra'], **guard)
            return [Item((move & ~(31 << USCHED)) | (w & (31 << USCHED)) | wait, i, 'move')]
        if name in ('ldg__sImmOffset', 'ld__sImmOffset') and ops.get('sp2'):
            # What the load says to the cache about its neighbours: sm_70 has no such thing to say.
            return [Item(self.put(d.c, w, 'sp2', 0), i, 'without its cache hint')]
        if (name == 'ldg__sImmOffset' and mc.decode(70, w) is None
                and ops.get('e') == 1 and ops.get('sz') in (5, 6)
                and ops.get('sem') in (0, 1, 2) and ops.get('sco') in (2, 3)
                and ops.get('cop') == 1):
            # The Turing table's LTC128B and the like: rebuild the load with sm_70's table, keeping
            # its data, guard and scoreboards. What changes is the cache hint.
            target = mc.assemble('ldg__sImmOffset',
                e=ops.get('e', 1), sz=ops['sz'], sem=ops['sem'], sco=ops['sco'], cop=ops['cop'],
                Pu=ops.get('Pu', 7), Rd=ops['Rd'], Ra=ops['Ra'], Ra_offset=ops.get('Ra_offset', 0),
                Pg=ops.get('Pg', 7), Pg_not=ops.get('Pg@not', 0),
                src_rel_sb=ops.get('src_rel_sb', 7), dst_wr_sb=ops.get('dst_wr_sb', 7))
            if mc.decode(70, target) is not None:
                return [Item(target, i, 'without its cache hint')]
        if name in mc.rule_of:
            out = self.rule(d, mc.rule_of[name], i)
            after = mc.decode(70, out)
            def uniform(operand):
                # Whether `operand` of the result stands for a uniform operand of the instruction:
                # the register that stands for it in the whole function, or the one it was loaded
                # into from the frame for this instruction.
                back = ordinary_back(operand, ops)
                return back != operand and (ops.get(back) in self.place or ops.get(back) in self._mem_regs)
            if after is not None and after.kind in MC.LATE and any(
                    letter == 'R' and operand in after.ops and uniform(operand)
                    for letter, operand, span in mc._uses[(70, after.c.name)].reads):
                # It reads a uniform register, and reads its operands late (`late`).
                return [self.late(after.c, out, i, 'ordinary registers')]
            return [Item(out, i, 'ordinary registers')]
        raise Refused('no translation for %s (%s) at 0x%x' % (d.c.name, d.mnemonic, 16 * i))

    # ---- Turing's matrix instructions ---------------------------------------------------------------------------
    # Volta has neither. What stands for them is what the layer already does in program text
    # (crates/vf-ptx/src/lower/ says why each sequence is right, and tcrun checks every
    # register of every thread against a model), as the compiler writes it for sm_70.
    #
    # A lane is 4 * group + thread (group 0 to 7, thread 0 to 3). The instructions of a sequence
    # that take their time (shuffles, loads, products) say on scoreboard SEQUENCE when their
    # result is there, and what reads it waits for that: by the end of a sequence each has
    # written, and so has read, and its registers are free for the next one.
    def matrix_load(self, i, d, guard, wait):
        """LDSM (`ldmatrix`): each thread gets one word of each of 1, 2 or 4 matrices of 8 rows of 8
        halves in shared memory. The address of row r of matrix m is what thread r + 8m has in the
        address register. Plain: the thread's pair of halves in the row of its group. Transposed:
        the halves of column `group` in the rows 2 * thread and 2 * thread + 1.

        The address comes by a shuffle from the lane that holds it, into the destination itself
        (the one destination that is the address register is shuffled last: what is issued before
        has read by then), and plain loads follow.

        A store before the load may not have read a destination yet: a store reads its registers
        some time after it is issued, and the compiler counts on the load, a memory instruction
        like the store, to write them after that (native code writes such a register with
        arithmetic only after a barrier or a wait for a later memory instruction). So the first
        instruction of the sequence that writes a destination is a shuffle, and arithmetic writes
        one only after waiting for the shuffles: the lanes of the rows are computed in registers
        of the sequence, and in a transposed load those of the first rows go into the
        destinations after the shuffles of the second rows are there. (Until 2026-11-23 the lanes
        went into the destinations first, and a store of such a register just before the load
        could store the lane.)"""
        mc, ops, w, name = self.mc, d.ops, d.word, d.c.name
        asm = mc.assemble
        if ops['sz'] != 0 or ops['mode'] not in (0, 1) or ops['num'] not in (0, 1, 2):
            raise Refused('this form of LDSM at 0x%x' % (16 * i))
        count, transposed = 1 << ops['num'], ops['mode'] == 1
        rd, ra, offset = ops['Rd'], ops['Ra'], ops.get('Ra_offset', 0)
        if rd == 255 or rd + count > 255:
            raise Refused('LDSM into no register at 0x%x' % (16 * i))
        uniform = None
        if name == 'ldsm__URsIR':
            uniform = self.R(ops['Ra_URb'])
            uniform = None if uniform == 255 else uniform
        lane, s = self.lane(), self.scratch()
        part, group, other, sum_ = s, s + 1, s + 2, s + 3            # then s + 4 to s + 7: the second rows of a transposed load
        result = ops['dst_wr_sb']
        new = []
        def add(word, what, waits=False):
            new.append(Item(word | wait | ((1 << (WAIT + SEQUENCE)) if waits else 0), i, what))
        shf = lambda **k: asm('shf__RuIR_RIR', fmt='U32', **k)
        lut = lambda **k: asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, **k)
        add(shf(Rd=group, Ra=255, Sb=2, Rc=lane, dir='R', hilo='HI'), 'group')
        # The one destination that is the address register comes last, and has its lane in a scratch register.
        order = [m for m in range(count) if rd + m != ra] + [m for m in range(count) if rd + m == ra]
        # What writes a register of the program keeps the guard, and so does what reads memory.
        # (The guard is taken to be the same for the threads that give addresses and those that
        # load: a thread that does not run a shuffle may not give its register to the others.)
        guarded = lambda r: guard if rd <= r < rd + count else {}
        if not transposed:
            add(lut(Rd=part, Ra=lane, Sb=3, Rc=255, imm8=0xc0), 'thread')
            # The lane of the row: group + 8m, in a register of the sequence.
            lanes = {}
            for m in order:
                if m == 0:
                    lanes[m] = group
                else:
                    lanes[m] = s + 4 + m
                    add(asm('iadd3_imm__RsIR_RIR', Rd=lanes[m], Ra=group, Sb=8 * m, Rc=255), 'row')
            for m in order:
                add(asm('shfl__RRI', shflmd=0, Pu=7, Rd=rd + m, Ra=ra, Rb=lanes[m], Sc=31, dst_wr_sb=SEQUENCE, **guard), 'address of the row')
            # The pair of the thread in its row: 4 * thread further (and the uniform part of the address).
            if uniform is not None:
                add(asm('lea_lo_noimm__RRR_RRR', Rd=sum_, Ra=part, Rb=uniform, scaleU5=2), 'place in the row')
            for k, m in enumerate(order):
                if uniform is None:
                    word = asm('lea_lo_noimm__RRR_RRR', Rd=rd + m, Ra=part, Rb=rd + m, scaleU5=2, **guard)
                else:
                    word = asm('iadd3_noimm__RRR_RRR', Rd=rd + m, Ra=rd + m, Rb=sum_, Rc=255, **guard)
                add(word, 'address', waits=k == 0)
            for m in order:
                add(asm('lds__sImmOffset', sz='U.32', Rd=rd + m, Ra=rd + m, Ra_offset=offset,
                        dst_wr_sb=SEQUENCE if result == 7 else result, **guard), 'word of the matrix')
            if result == 7:
                new[-1].then = SEQUENCE
        else:
            # Twice the thread: the first of its two rows.
            add(shf(Rd=part, Ra=lane, Sb=1, Rc=255, dir='L', hilo='LO'), 'thread')
            add(lut(Rd=part, Ra=part, Sb=6, Rc=255, imm8=0xc0), 'thread')
            first, second = {}, {}
            for m in order:
                second[m] = s + 4 + m
                add(asm('iadd3_imm__RsIR_RIR', Rd=second[m], Ra=part, Sb=8 * m + 1, Rc=255), 'row')
            for m in order:
                add(asm('shfl__RRI', shflmd=0, Pu=7, Rd=second[m], Ra=ra, Rb=second[m], Sc=31, dst_wr_sb=SEQUENCE), 'address of the row')
            # The lanes of the first rows, in the destinations once the shuffles above are there.
            waits = True
            for m in order:
                if m == 0:
                    first[m] = part
                else:
                    first[m] = other if rd + m == ra else rd + m
                    add(asm('iadd3_imm__RsIR_RIR', Rd=first[m], Ra=part, Sb=8 * m, Rc=255, **guarded(first[m])), 'row', waits=waits)
                    waits = False
            for m in order:
                add(asm('shfl__RRI', shflmd=0, Pu=7, Rd=rd + m, Ra=ra, Rb=first[m], Sc=31, dst_wr_sb=SEQUENCE, **guard), 'address of the row')
            # The half of the thread's column in each row: 2 * group further (and the uniform part).
            if uniform is not None:
                add(asm('lea_lo_noimm__RRR_RRR', Rd=sum_, Ra=group, Rb=uniform, scaleU5=1), 'place in the row')
            for k, m in enumerate(order):
                for r, g in ((second[m], {}), (rd + m, guard)):
                    if uniform is None:
                        word = asm('lea_lo_noimm__RRR_RRR', Rd=r, Ra=group, Rb=r, scaleU5=1, **g)
                    else:
                        word = asm('iadd3_noimm__RRR_RRR', Rd=r, Ra=r, Rb=sum_, Rc=255, **g)
                    add(word, 'address', waits=k == 0 and r == second[m])
            for m in order:
                # (Under the guard too, though it writes a register of the translation: it reads
                # memory, which the instruction does not when its guard is false. Until
                # 2026-11-17 it had none.)
                add(asm('lds__sImmOffset', sz='U.U16', Rd=second[m], Ra=second[m], Ra_offset=offset, dst_wr_sb=SEQUENCE, **guard), 'half of the matrix')
                add(asm('lds__sImmOffset', sz='U.U16', Rd=rd + m, Ra=rd + m, Ra_offset=offset, dst_wr_sb=SEQUENCE, **guard), 'half of the matrix')
            # The two halves in one word: the one of the second row above the other.
            for k, m in enumerate(order):
                add(asm('lea_lo_noimm__RRR_RRR', Rd=rd + m, Ra=second[m], Rb=rd + m, scaleU5=16, **guard), 'word of the matrix', waits=k == 0)
        new[-1].word = (new[-1].word & ~(31 << USCHED)) | (w & (31 << USCHED))
        return new

    def matrix_product(self, i, d, wait):
        """HMMA.1688 (`mma.m16n8k8` on halves): D[16x8] = A[16x8] * B[8x8] + C over the warp. Volta's
        own instruction makes four products of 8x8x4 a warp, one for each eight threads, in four
        steps (or two) each. Two of them do it, one for the even k and one for the odd:

        - as an operand held by columns, the two registers of A are taken as they are, then with
          their halves swapped;
        - of B a thread gives its eight threads four elements of one k for four columns: the
          lower half of the warp columns 0, 1, 4, 5 with the even k, the upper half columns 2, 3,
          6, 7 with the odd k (and the other way round the second time): one shuffle brings the
          neighbour's register (the lane 4 further), four more the words from the lanes that
          have them;
        - with accumulators in floats, a thread finds its four results in the first four of the
          eight registers that Volta's instruction writes when its number is even, in the last
          four when it is odd.

        The lanes that hold what a thread gathers and the two masks are those the function
        keeps (`kept`). The four registers of the gathered B are kept too: a product on the same
        register of B as the last one that gathered, with nothing since that writes it and no
        other way to it, does not gather again (`kept_gathering`). In floats: 25 instructions
        when it gathers, 14 when it does not (33 every time until 2026-11-18).

        At level 3 (everything at the instruction): the lane is read there, low/high/half/odd
        are computed there, B is gathered every time, and every register (scratch, gathered,
        lane values) is taken at the instruction. 16 registers at once: 1 lane + 3 for
        low/high/half + 8 scratch + 4 gathered; odd is computed after gathering into the
        register that held half, since low/high/half are dead once B is gathered
        (docs/translator.md, "Registers")."""
        mc, ops, w = self.mc, d.ops, d.word
        asm = mc.assemble
        if ops['Pg'] != 7 or ops.get('Pg@not'):
            raise Refused('HMMA.1688 under a predicate at 0x%x' % (16 * i))
        if ops.get('Ra@negate') or ops.get('Rb@negate') or ops.get('satfinite') or ops.get('size') or ops.get('altfmt') or ops['dstfmt'] not in (0, 1):
            raise Refused('this form of HMMA.1688 at 0x%x' % (16 * i))
        floats = ops['dstfmt'] == 1                    # accumulators in floats (four registers), or in halves (two)
        rd, ra, rb, rc = ops['Rd'], ops['Ra'], ops['Rb'], ops['Rc']
        # (The zero register for the accumulators: a product with nothing to add to. Volta's
        # instruction takes it too. In floats only: in halves the accumulators go through an
        # exchange first, which nothing would be needed for; no code at hand has that form.)
        zero = rc == 255
        if 255 in (rd, ra, rb) or rd % 2 or ra % 2 or (rc % 2 and not zero) or (zero and not floats):
            raise Refused('HMMA.1688 on these registers at 0x%x' % (16 * i))
        d_set, c_set = set(range(rd, rd + 4)), set() if zero else set(range(rc, rc + 4))
        # D and B may share a register: B is gathered into separate registers before the HMMA
        # steps, and nothing of the product's sequence reads B after gathering. When the product
        # does not gather (kept_gathering: B was gathered by the last one), it does not read rb
        # at all; the next product on that rb gathers again, because kept_gathering checks
        # self.src[gathered:i] and finds the HMMA instruction there, whose d.writes include rb
        # (rd+3 == rb). In a pair on whole rows (pairing.py), the HMMA steps read B from row
        # registers of the translation (self.half_row), not from the program's rb; the whole-row
        # load reads from shared memory using the address register (Ra), not rb. In
        # product_in_halves, the same applies: B is in the gathered registers (b1, b2). So D
        # overlapping B is harmless at every level and in every path.
        if floats and ((rd != rc and d_set & c_set) or d_set & {ra, ra + 1}):
            raise Refused('HMMA.1688 that writes over its operands at 0x%x' % (16 * i))
        # Level 3: everything taken at the instruction. The lane is read there, the four values
        # (low, high, half, odd) computed there, B is gathered every time.
        level3 = self.borrow is not None and self.borrow.everything
        if level3:
            lane = self.lane()          # taken at the instruction, S2R issued by translate()
            low = self.take(1)
            high = self.take(1)
            half = self.take(1)
            # The scratch registers of the product, taken by their alignment need instead of
            # as one block of 8: room can then be made in small functions where 8 consecutive
            # registers are not available. sums (two pairs from even, used as HMMA Rd and as
            # gathering temps; not necessarily contiguous), swapped (2 from even, used as HMMA
            # Ra), pair (1, halves only), gathered B (two pairs from even).
            sums_lo = self.take(2, 'product sums 0-1')   # HMMA Rd of steps 0-1 and gathering temps
            sums_hi = self.take(2, 'product sums 2-3')   # HMMA Rd of steps 2-3
            swapped = self.take(2, 'product swapped')
            pair = self.take(1, 'product pair') if not floats else 0
            b1 = self.take(2, 'product gathered B1')
            b2 = self.take(2, 'product gathered B2')
            # Names for the existing code: sums is sums_lo (the first pair), and the four
            # individual sum registers are s0, s1, s2, s3.
            sums = sums_lo
            s0, s1, s2, s3 = sums_lo, sums_lo + 1, sums_hi, sums_hi + 1
            near, even, other = s0, s1, s2
        else:
            if self.kept_at is None or self.gathered_at is None:
                raise Refused('a product in a function that was not given the registers it keeps')
            s = self.scratch()
            low, high, half, odd = range(self.kept_at, self.kept_at + KEPT)
            b1, b2 = self.gathered_at, self.gathered_at + 2
            # Of the scratch registers: the four of the sums, three of which first hold what is
            # being gathered of B (the first step of a product writes them, and waits for the
            # shuffles that read them); A with its halves swapped; and, for accumulators in
            # halves, all ones in the threads 2 and 3 of a group.
            sums, swapped, pair = s, s + 4, s + 6
            near, even, other = sums, sums + 1, sums + 2
        s_all = (s0, s1, s2, s3) if level3 else (sums, sums + 1, sums + 2, sums + 3)
        new = []
        def add(word, what, waits=False):
            new.append(Item(word | wait | ((1 << (WAIT + SEQUENCE)) if waits else 0), i, what))
        shf = lambda **k: asm('shf__RuIR_RIR', **{'fmt': 'U32', **k})
        choose = lambda **k: asm('lop3_lut__RRR_RRR', Pu=7, Pp=7, Pp_not=1, imm8=0xe4, **k)        # Rc ? Ra : Rb, bit by bit
        def halves_swapped():
            for k in range(2):
                add(shf(Rd=swapped + k, Ra=ra + k, Sb=16, Rc=ra + k, dir='L', cw='W', hilo='HI'), 'halves swapped')
        if level3:
            # At level 3: compute low, high, half from lane (the same instructions as `kept`),
            # then gather B (always), then compute odd into the register that held half (dead
            # after gathering: its last use is the two choose operations before the final shuffles
            # to b1..b2+1, which read near, even, low, high but not half).
            add(shf(Rd=low, Ra=255, Sb=1, Rc=lane, dir='R', hilo='HI'), 'kept: low')
            add(asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, Rd=low, Ra=low, Sb=8, Rc=255, imm8=0xc0), 'kept: low')
            add(asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, Rd=low, Ra=low, Sb=3, Rc=lane, imm8=0xf8), 'kept: low')
            add(asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, Rd=high, Ra=low, Sb=16, Rc=255, imm8=0xfc), 'kept: high')
            add(shf(Rd=half, Ra=lane, Sb=28, Rc=255, dir='L', hilo='LO'), 'kept: half')
            add(shf(Rd=half, Ra=255, Sb=31, Rc=half, dir='R', hilo='HI', fmt='S32'), 'kept: half')
            # Gather B.
            add(asm('shfl__RII', shflmd=3, Pu=7, Rd=near, Ra=rb, Sb=4, Sc=31, dst_wr_sb=SEQUENCE), 'neighbour')
            add(shf(Rd=even, Ra=rb, Sb=16, Rc=255, dir='L', hilo='LO'), 'columns')
            halves_swapped()
            add(shf(Rd=other, Ra=255, Sb=16, Rc=near, dir='R', hilo='HI'), 'columns', waits=True)
            add(shf(Rd=even, Ra=even, Sb=16, Rc=near, dir='R', cw='W', hilo='LO'), 'columns')
            add(shf(Rd=other, Ra=rb, Sb=16, Rc=other, dir='L', cw='W', hilo='HI'), 'columns')
            add(choose(Rd=near, Ra=other, Rb=even, Rc=half), 'columns')
            add(choose(Rd=even, Ra=even, Rb=other, Rc=half), 'columns')
            for to, value, lanes in ((b1, near, low), (b1 + 1, near, high), (b2, even, low), (b2 + 1, even, high)):
                add(asm('shfl__RRI', shflmd=0, Pu=7, Rd=to, Ra=value, Rb=lanes, Sc=31, dst_wr_sb=SEQUENCE), 'gathered')
            # low, high, half are dead. Compute odd into the register that held half.
            odd = half
            add(shf(Rd=odd, Ra=lane, Sb=31, Rc=255, dir='L', hilo='LO'), 'kept: odd')
            add(shf(Rd=odd, Ra=255, Sb=31, Rc=odd, dir='R', hilo='HI', fmt='S32'), 'kept: odd')
        elif self.kept_gathering(i, rb):
            halves_swapped()
        else:
            # The elements (2k, c), (2k, c + 1) and (2k + 1, c), (2k + 1, c + 1) of B, in the lanes of an even group c.
            add(asm('shfl__RII', shflmd=3, Pu=7, Rd=near, Ra=rb, Sb=4, Sc=31, dst_wr_sb=SEQUENCE), 'neighbour')
            add(shf(Rd=even, Ra=rb, Sb=16, Rc=255, dir='L', hilo='LO'), 'columns')
            halves_swapped()
            add(shf(Rd=other, Ra=255, Sb=16, Rc=near, dir='R', hilo='HI'), 'columns', waits=True)
            add(shf(Rd=even, Ra=even, Sb=16, Rc=near, dir='R', cw='W', hilo='LO'), 'columns')
            add(shf(Rd=other, Ra=rb, Sb=16, Rc=other, dir='L', cw='W', hilo='HI'), 'columns')
            # The upper half gives the other parity: first the odd k there, then the even.
            add(choose(Rd=near, Ra=other, Rb=even, Rc=half), 'columns')
            add(choose(Rd=even, Ra=even, Rb=other, Rc=half), 'columns')
            for to, value, lanes in ((b1, near, low), (b1 + 1, near, high), (b2, even, low), (b2 + 1, even, high)):
                add(asm('shfl__RRI', shflmd=0, Pu=7, Rd=to, Ra=value, Rb=lanes, Sc=31, dst_wr_sb=SEQUENCE), 'gathered')
            self.holds = (i, rb)
        if not floats:
            return self.product_in_halves(i, d, wait, new, add, choose, shf, (rd, ra, rc),
                                          (pair, odd, b1, b2, swapped, sums, s_all if level3 else None),
                                          level3=level3, lane=lane if level3 else None)
        def step(n, to, a, b, c, **more):
            word = asm('hmma__884_f32', size=0, dstfmt=1, srcfmt=1, step=n, satfinite=0, Rd=to, Ra=a, tnspA=1, Rb=b, tnspB=0, Rc=c, **more)
            return with_stall(word, 2)
        # Registers 0 to 3 of Volta's eight in scratch registers, 4 to 7 in the destination. (Each
        # step says when its result is there: the compiler says it of the last one only, and
        # keeps 32 cycles to a product that reads another's result where it does not wait.)
        # As the compiler has them: the steps two cycles apart; the last one of a product says
        # when its result is there, and the first of the next waits for that; after the last
        # one of all, eight cycles before anything else reads a result (a wait alone is not
        # enough: with every step naming the scoreboard and the selections two cycles after the
        # last, a V100 selects from registers that are not written yet).
        upper = rc if zero else rc + 2
        # (At level 3, sums may not be contiguous: s0..s3 name the four registers.)
        s_lo, s_hi = (s0, s2) if level3 else (sums, sums + 2)
        for n, (to, c) in enumerate(((s_lo, rc), (s_hi, upper), (rd, rc), (rd + 2, upper))):
            add(step(n, to, ra, b1, c, **({'dst_wr_sb': SEQUENCE} if n == 3 else {})), 'product', waits=n == 0)
            new[-1].covered = True
        for n, to in enumerate((s_lo, s_hi, rd, rd + 2)):
            add(step(n, to, swapped, b2, to, **({'dst_wr_sb': SEQUENCE, 'src_rel_sb': SEQUENCE} if n == 3 else {})), 'product', waits=n == 0)
            new[-1].covered = True
        new[-1].word = with_stall(new[-1].word, 8)
        s_all = (s0, s1, s2, s3) if level3 else tuple(sums + k for k in range(4))
        for k in range(4):
            add(choose(Rd=rd + k, Ra=rd + k, Rb=s_all[k], Rc=odd), 'the thread\'s own', waits=k == 0)
        new[-1].word = (new[-1].word & ~(31 << USCHED)) | (w & (31 << USCHED))
        return new

    def product_in_halves(self, i, d, wait, new, add, choose, shf, registers, scratch, level3=False, lane=None):
        """The end of `matrix_product` for accumulators in halves (HMMA.1688.F16): Volta's
        instruction then writes four registers of two halves, and a thread holds one row of its
        eight threads' product: in registers 2 * (thread & 1) and the next, what Turing's layout
        puts in register thread >> 1 of the threads thread & 1 and (thread & 1) + 2 of the group:
        one register of its own and one of the thread two lanes away. One exchange with that
        thread before, one after.
        At level 3: `pair` is computed from `lane` here (after gathering: the register that
        held `half` now holds `odd`, and `pair` goes into the scratch register `s+6`)."""
        mc, w, asm = self.mc, d.word, self.mc.assemble
        if lane is None:
            lane = self.lane()
        (rd, ra, rc), (pair, odd, b1, b2, swapped, sums, s_a) = registers, scratch
        # (At level 3, s_a is the four sum registers; otherwise None and sums+k is used.)
        sa = s_a if s_a else (sums, sums + 1, sums + 2, sums + 3)
        # All ones in the threads 2 and 3 of each group (bit 1 of the lane).
        add(shf(Rd=pair, Ra=lane, Sb=30, Rc=255, dir='L', hilo='LO'), 'pair')
        add(shf(Rd=pair, Ra=255, Sb=31, Rc=pair, dir='R', hilo='HI', fmt='S32'), 'pair')
        # The accumulator this thread holds for the other one, sent; then its row of the eight threads' matrix.
        add(choose(Rd=sa[2], Ra=rc, Rb=rc + 1, Rc=pair), 'for the other')
        add(asm('shfl__RII', shflmd=3, Pu=7, Rd=sa[3], Ra=sa[2], Sb=2, Sc=31, dst_wr_sb=SEQUENCE), 'exchange')
        add(choose(Rd=sa[0], Ra=sa[3], Rb=rc, Rc=pair), 'row', waits=True)
        add(choose(Rd=sa[1], Ra=rc + 1, Rb=sa[3], Rc=pair), 'row')
        add(choose(Rd=sa[2], Ra=sa[3], Rb=rc, Rc=pair), 'row')
        add(choose(Rd=sa[3], Ra=rc + 1, Rb=sa[3], Rc=pair), 'row')
        def step(n, a, b, **more):
            word = asm('hmma__884_f16_f16', size=0, dstfmt=0, srcfmt=0, step=n, satfinite=0, Rd=sa[2 * n], Ra=a, tnspA=1, Rb=b, tnspB=0, Rc=sa[2 * n], **more)
            return with_stall(word, 4)
        # As the compiler has them: four cycles apart, thirteen after the last (`matrix_product`).
        for n in range(2):
            add(step(n, ra, b1, **({'dst_wr_sb': SEQUENCE} if n == 1 else {})), 'product', waits=n == 0)
            new[-1].covered = True
        for n in range(2):
            add(step(n, swapped, b2, **({'dst_wr_sb': SEQUENCE, 'src_rel_sb': SEQUENCE} if n == 1 else {})), 'product', waits=n == 0)
            new[-1].covered = True
        new[-1].word = with_stall(new[-1].word, 13)
        # Back: the row of the thread, what it holds for the other one sent, and its own two registers.
        add(choose(Rd=sa[0], Ra=sa[2], Rb=sa[0], Rc=odd), 'row', waits=True)
        add(choose(Rd=sa[1], Ra=sa[3], Rb=sa[1], Rc=odd), 'row')
        add(choose(Rd=sa[2], Ra=sa[0], Rb=sa[1], Rc=pair), 'for the other')
        add(asm('shfl__RII', shflmd=3, Pu=7, Rd=sa[3], Ra=sa[2], Sb=2, Sc=31, dst_wr_sb=SEQUENCE), 'exchange')
        add(choose(Rd=rd, Ra=sa[3], Rb=sa[0], Rc=pair), 'the thread\'s own', waits=True)
        add(choose(Rd=rd + 1, Ra=sa[1], Rb=sa[3], Rc=pair), 'the thread\'s own')
        new[-1].word = (new[-1].word & ~(31 << USCHED)) | (w & (31 << USCHED))
        return new

    # ---- pairs of products on whole rows (docs/pairs.md) ---------------------------------------------
    def kept_for_rows(self):
        """What a function with pairs computes of the lane when it is entered, each into a register
        that nothing else writes (`placement`). With the bits of the lane L = b4 b3 b2 b1 b0:
        kA = b1 b3 b4 b0 (the row of A a thread loads; kA2: of the second fragment, 16 further),
        kB = b4 b2 b1 b0 (of a plain B; kB2: of the second pair), kBT = b4 0 b1 b0 and cB = b2
        000 (of a transposed B: its row, and the byte in it), pi = b3 b4 b0 b2 b1 (the lane
        whose Turing register a thread holds) and inv, the other way round: b3 b4 b1 b0 b2 read
        as the lane that holds a thread's."""
        asm, lane, t, kept = self.mc.assemble, self.lane(), self.scratch(), self.kept_rows
        shf = lambda **k: asm('shf__RuIR_RIR', **{'fmt': 'U32', **k})
        right = lambda to, x, n: shf(Rd=to, Ra=255, Sb=n, Rc=x, dir='R', hilo='HI')
        left = lambda to, x, n: shf(Rd=to, Ra=x, Sb=n, Rc=255, dir='L', hilo='LO')
        masked = lambda to, x, m: asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, Rd=to, Ra=x, Sb=m, Rc=255, imm8=0xc0)
        both = lambda to, a, b: asm('lop3_lut__RRR_RRR', Pu=7, Pp=7, Pp_not=1, imm8=0xfc, Rd=to, Ra=a, Rb=b, Rc=255)
        with_low = lambda to, x, m: asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, Rd=to, Ra=x, Sb=m, Rc=lane, imm8=0xf8)   # x | (lane & m)
        words = []
        for name in ROW_KEPT:
            if name not in kept or name not in self.q.kept():
                continue
            r = kept[name]
            if name == 'kA':
                words += [left(r, lane, 2), masked(r, r, 8), right(t, lane, 1), masked(t, t, 4), both(r, r, t),
                          right(t, lane, 3), masked(t, t, 2), both(r, r, t), with_low(r, r, 1)]
            elif name in ('kB', 'kBT'):
                words += [right(r, lane, 1), masked(r, r, 8), with_low(r, r, 7 if name == 'kB' else 3)]
            elif name in ('kA2', 'kB2'):
                words += [asm('lop3_lut__RuIR_RIR', Pu=7, Pp=7, Pp_not=1, Rd=r, Ra=kept[name[:2]], Sb=16, Rc=255, imm8=0xfc)]
            elif name == 'cB':
                words += [left(r, lane, 1), masked(r, r, 8)]
            elif name == 'pi':
                words += [right(r, lane, 1), masked(r, r, 0xb), left(t, lane, 1), masked(t, t, 16), both(r, r, t),
                          left(t, lane, 2), masked(t, t, 4), both(r, r, t)]
            else:
                words += [left(r, lane, 1), masked(r, r, 0x16), right(t, lane, 1), masked(t, t, 8), both(r, r, t),
                          right(t, lane, 2), masked(t, t, 1), both(r, r, t)]
        return [Item(word, 0, 'kept') for word in words]

    def half_row(self, rd, q, half):
        """The first of the two registers that hold half `half` of k of the row of fragment or
        pair `q` of the whole-row loads with destination `rd` (docs/pairs.md, section 4)."""
        if rd not in self.rows_at:
            raise Refused('rows of a load in a function that was not given registers for them')
        if self.rows_n[rd] == 4:
            return rd + 2 * q if half == 0 else self.rows_at[rd] + 2 * q
        return self.rows_at[rd] + 2 * q + 4 * half

    def whole_rows(self, i, d, guard, wait):
        """A matrix load whose results the products of pairs read: each thread loads whole rows
        of 16 bytes, one for each of the two fragments of A or pairs of B (two, of 8 bytes, for
        each pair of a transposed B), at the address that the lane named by a kept value holds:
        one shuffle of the address register a row, into a scratch register, then the loads.

        When what the load gives in Turing's layout may be read by something else (`also`), the
        sequence of `matrix_load` follows, into the program's registers; the rows are then in
        registers of the translation only."""
        mc, ops, w, name = self.mc, d.ops, d.word, d.c.name
        asm = mc.assemble
        load = self.q.loads[i]
        role, rd, ra, offset = load['role'], load['rd'], ops['Ra'], ops.get('Ra_offset', 0)
        if guard.get('Pg', 7) != 7 or ops['sz'] != 0 or ops['num'] != 2:
            raise Refused('this form of LDSM for whole rows at 0x%x' % (16 * i))
        uniform = None
        if name == 'ldsm__URsIR':
            uniform = self.R(ops['Ra_URb'])
            uniform = None if uniform == 255 else uniform
        s = self.scratch()
        result = ops['dst_wr_sb']
        new = []
        # (`through`: the rows' addresses come from the companions of a root register, as they
        # are or by the instruction that makes the address register, done again on them.)
        through = self.q.through.get(i)
        # (The shuffles of the addresses say on the scoreboard ARRIVES when they are there, and
        # the loads wait for that one and say on it when they have read: not on the scoreboard
        # the products name, or every load would wait for the products before it and everything
        # after a load for them too. When the program's load named ARRIVES itself for its
        # result, the other one serves.)
        board = SEQUENCE if result == ARRIVES else ARRIVES
        def add(word, what, waits=False):
            # (What the program's load waited for, the first instruction here waits for: on the
            # others it would make each load of a row wait for the one before it, when they
            # name a scoreboard the program's load waited for.)
            new.append(Item(word | (0 if new else wait) | ((1 << (WAIT + board)) if waits else 0), i, what))
        # The rows: (the scratch register of its address, the register that holds the lane which
        # has it, or the kept one and how much further, the registers of its words: pairs, and
        # the byte each is at). The shuffles first, on the address register as it is; what is
        # added to an address (the uniform part; the byte of a transposed B) after them: an
        # addition before a shuffle keeps it eight cycles (profiled).
        if role in ('A', 'B'):
            keys = [self.kept_rows['kA' if role == 'A' else 'kB'], self.kept_rows['kA2' if role == 'A' else 'kB2']]
            rows = [(s + q, keys[q], 0, [(self.half_row(rd, q, 0), 0), (self.half_row(rd, q, 1), 8)]) for q in range(2)]
        else:
            rows = [(s + 2 * q + half, self.kept_rows['kBT'], 16 * q + 4 * half, [(self.half_row(rd, q, half), 0)]) for q in range(2) for half in range(2)]
        spare, lanes = s + 4, []
        for at, key, further, words in rows:
            if further and through is None:
                add(asm('iadd3_imm__RsIR_RIR', Rd=spare, Ra=key, Sb=further, Rc=255), 'whole row: lane')
                key, spare = spare, spare + 1
            lanes.append(key)
        more = uniform
        if role == 'BT':
            more = self.kept_rows['cB']
            if uniform is not None:
                more = s + 7
                add(asm('iadd3_noimm__RRR_RRR', Rd=more, Ra=uniform, Rb=self.kept_rows['cB'], Rc=255), 'whole row: address')
        first = True
        if through is None:
            for (at, key, further, words), held in zip(rows, lanes):
                add(asm('shfl__RRI', shflmd=0, Pu=7, Rd=at, Ra=ra, Rb=held, Sc=31, dst_wr_sb=board), 'whole row: address')
            if more is not None:
                for at, key, further, words in rows:
                    add(asm('iadd3_noimm__RRR_RRR', Rd=at, Ra=at, Rb=more, Rc=255), 'whole row: address', waits=first)
                    first = False
        else:
            # No shuffle: each row's address is the root's companion for that row (made where
            # the plan has it, long arrived), through the instruction that makes the address
            # register when the root is another register, plus what is added to it. The
            # address is then in the companion itself, or in the row's scratch register.
            root, made = through
            held = [self.companion(root, role, k) for k in range(len(rows))]
            if made is not None:
                for k, (at, key, further, words) in enumerate(rows):
                    add(self.again(made, root, held[k], at), 'whole row: address')
                    held[k] = at
            if more is not None:
                for k, (at, key, further, words) in enumerate(rows):
                    add(asm('iadd3_noimm__RRR_RRR', Rd=at, Ra=held[k], Rb=more, Rc=255), 'whole row: address')
                    held[k] = at
            rows = [(held[k], key, further, words) for k, (at, key, further, words) in enumerate(rows)]
            first = False
        for at, key, further, words in rows:
            for to, byte in words:
                for k in range(1 if ALIGNED_ROWS else 2):
                    word = asm('lds__sImmOffset', sz='U.64' if ALIGNED_ROWS else 'U.32', Rd=to + k, Ra=at, Ra_offset=offset + byte + 4 * k,
                               dst_wr_sb=SEQUENCE if result == 7 else result, src_rel_sb=board)
                    # (The loads read their address, a scratch register, some time after they
                    # are issued: each says when it has read, and what follows the last waits
                    # for that. Nothing waits here for what they load: the products that read
                    # it wait, for the scoreboard the program gave the load, or for the
                    # sequences' own, which every half of a pair waits for.)
                    add(word & ~REUSE, 'whole row', waits=first)
                    first = False
        new[-1].late = True
        if load['also']:
            # What the program's registers get, as before: it reads the address register after
            # the rows did, and its scratch registers when the loads above have read theirs.
            old = self.matrix_load(i, d, guard, wait)
            old[0].word |= 1 << (WAIT + ((new[-1].word >> READ_BARRIER) & 7))
            new[-1].late = False
            return new + old
        # (As late after the load as in the original: what its instructions have not already taken.)
        taken = sum(stall_of(item.word) for item in new[:-1])
        new[-1].word = with_stall(new[-1].word, max(stall_of(new[-1].word), stall_of(w) - taken, 1))
        return new

    def companion(self, root, role, k):
        """The register that holds the root register as the lane of row `k` of a load of `role` has it."""
        if root not in self.comp_at or role not in self.q.roots.get(root, ()):
            raise Refused('a companion in a function that was not given registers for it')
        at = self.comp_at[root]
        for r in self.q.roots[root]:
            if r == role:
                return at + k
            at += pairing.ROLE_ROWS[r]

    def again(self, made, root, held, to):
        """The instruction `made` of the program, which makes an address register from the register
        `root` and constants, done on `held` in the place of the root, into `to`: its schedule
        left to `schedule`, like any instruction of the translation."""
        d = self.src[made]
        same = self.mc.decode(70, d.word)
        if same is None:
            raise Refused('the instruction at 0x%x is not one to do again on a companion' % (16 * made))
        c = self.mc.md70.by_name[same.c.name]
        word = self.put(c, d.word, 'Rd', to)
        for operand in ('Ra', 'Rb', 'Rc'):
            if d.ops.get(operand) == root:
                word = self.put(c, word, operand, held)
        return with_stall(word & ~(63 << WAIT) & ~REUSE, 1)

    def accompanied(self, i, plan):
        """The companions that `plan` (before or after instruction i) has made there: for each
        root and each row of each of its roles, the root as the lane of that row has it, by one
        shuffle, into a register nothing else writes. As for conversions: first an instruction
        that does nothing, waits for every scoreboard and stalls as long as one can (the root
        is read where no instruction of the program read it; and the loads that read the
        companions made before have read); what follows waits for the shuffles."""
        roots = plan.get(i)
        if not roots:
            return []
        asm, s = self.mc.assemble, self.scratch()
        out, shuffles = [Item(with_stall(NOP, 15) | (63 << WAIT), i, 'settled')], []
        for root in roots:
            for role in self.q.roots[root]:
                if role == 'BT':
                    lanes = [self.kept_rows['kBT']]
                    for k, further in enumerate((4, 16, 20)):
                        out.append(Item(asm('iadd3_imm__RsIR_RIR', Rd=s + 4 + k, Ra=self.kept_rows['kBT'], Sb=further, Rc=255), i, 'companion: lane'))
                        lanes.append(s + 4 + k)
                else:
                    lanes = [self.kept_rows['kA' if role == 'A' else 'kB'], self.kept_rows['kA2' if role == 'A' else 'kB2']]
                for k, lane in enumerate(lanes):
                    word = asm('shfl__RRI', shflmd=0, Pu=7, Rd=self.companion(root, role, k), Ra=root, Rb=lane, Sc=31, dst_wr_sb=SEQUENCE)
                    shuffles.append(Item(word | ((1 << (WAIT + SEQUENCE)) if shuffles and len(shuffles) % 8 == 0 else 0), i, 'companion'))
                # (The lanes of a transposed B are in scratch registers: its shuffles before the next role's sums.)
                out, shuffles = out + shuffles, []
        out[-1].then = SEQUENCE
        return out

    def half_of_pair(self, i, d, wait):
        """A product that is half of a pair: one of Volta's products of 8x8x4 on the half of k of
        the two rows that the pair's first product (the first half) or its second one (the
        second half) stands for, into the accumulators of both products of the pair: steps 0 and
        1 write those of the product on the even B register, steps 2 and 3 the other's. A by
        rows; B by columns, or by rows when it was loaded transposed. As the compiler has
        them: two cycles apart; the last step says when the result is there, and the first step
        of every such product waits for that scoreboard (measured: it costs nothing)."""
        mc, w = self.mc, d.word
        site = self.q.sites[i]
        a = self.half_row(site['ra'], site['a'], site['half'])
        b = self.half_row(site['rb'], site['p'], site['half'])
        new = []
        to = (site['x'], site['x'] + 2, site['y'], site['y'] + 2)
        # (What each step adds to: the first product of a pair starts from the accumulators the
        # two products of the program read, which may be other registers than their destinations,
        # or zero; the second goes on in place.)
        onto = to if site['half'] else tuple(255 if c is None else c + k for c in (site['cx'], site['cy']) for k in (0, 2))
        for n in range(4):
            word = mc.assemble('hmma__884_f32', size=0, dstfmt=1, srcfmt=1, step=n, satfinite=0, Rd=to[n], Ra=a, tnspA=0,
                               Rb=b, tnspB=0 if site['transposed'] else 1, Rc=onto[n], **({'dst_wr_sb': SEQUENCE} if n == 3 else {}))
            # (The program's waits and the wait for the products before: on the first step.)
            item = Item(with_stall(word, 2) | ((wait | (1 << (WAIT + SEQUENCE))) if n == 0 else 0), i, 'half of a pair')
            item.covered = n < 3
            new.append(item)
        # (What follows an instruction is as late after it as in the original: the four steps
        # take eight cycles, and the last one more when the original's stall was longer than that.)
        new[-1].word = with_stall(new[-1].word, max(2, stall_of(w) - 6))
        return new

    def converted(self, i, plan):
        """The accumulator registers that `plan` (before or after instruction i) has change state:
        each shuffled between lanes by the kept permutation, one way or the other. Before the
        first: an instruction that does nothing, waits for every scoreboard and stalls as long
        as an instruction can. A conversion reads its register where no instruction of the
        program did: whatever wrote it (a load whose reader, further on, has the wait; a
        product, after which what is not a product reads stale registers for eight cycles, wait
        or no wait; anything else, by its distance) is done by then. (It waited
        for the products only until 2026-11-18: accumulators loaded from memory were converted
        before they had arrived, and came out wrong on a V100.) Every eighth
        shuffle waits for those before it."""
        wanted = plan.get(i)
        if not wanted:
            return []
        out = [Item(with_stall(NOP, 15) | (63 << WAIT), i, 'settled')]
        for k, (r, to) in enumerate(wanted):
            word = self.mc.assemble('shfl__RRI', shflmd=0, Pu=7, Rd=r, Ra=r, Rb=self.kept_rows['pi' if to == pairing.Q else 'inv'], Sc=31, dst_wr_sb=SEQUENCE)
            out.append(Item(word | ((1 << (WAIT + SEQUENCE)) if k and k % 8 == 0 else 0), i, 'to pairs' if to == pairing.Q else 'from pairs'))
        out[-1].then = SEQUENCE
        return out

    def uniform_address(self, i, d, plain):
        """A memory instruction whose address has a uniform part, [Ra + UR + offset]: the sum of the
        two in the pool, then the plain instruction on it; or, when there is nothing else, the plain
        instruction on the ordinary registers of the uniform part."""
        mc, ops, w = self.mc, d.ops, d.word
        part = 'Ra_URb' if 'Ra_URb' in ops else 'Ra_URc'
        shared = plain in SHARED
        rule = mc.rule(d.c.name, plain, (part, 'stride', 'Ra') if shared else (part, 'sp2', 'TABLE', 'Ra'))
        if rule is None:
            raise Refused('no rule from %s to %s' % (d.c.name, plain))
        for key, fields in rule.dropped.items():
            # (What a load says to the cache about its neighbours, `sp2`, sm_70 has no way to say:
            # it is left out, as for a load at a plain address.)
            if key not in (part, 'Ra', 'stride', 'sp2') and any(f.get(w) for f in fields):
                raise Refused('%s at 0x%x asks for something sm_70 loads do not have (%s)' % (d.c.name, 16 * i, key))
        out = rule.apply(w, self.R, self.P)
        if out is None or ops[part] == 63:
            raise Refused('%s at 0x%x has a value that sm_70 does not have' % (d.c.name, 16 * i))
        target = mc.md70.by_name[plain]
        uniform = self.R(ops[part])
        wait = w & (63 << WAIT)
        if ops['Ra'] == 255:
            return [self.late(target, self.put(target, out, 'Ra', uniform), i, 'on the uniform part')]
        if shared:
            # In shared memory: 32 bits, the register times its stride plus the uniform part.
            scale = 1 << STRIDES[ops['stride']] if ops.get('stride') else 1
            add = mc.assemble('imad__RsIR_RIR', Rd=self.pool(1), Ra=ops['Ra'], Sb=scale, Rc=uniform)
            return [Item(add | wait, i, 'address'), self.late(target, self.put(target, out, 'Ra', self.pool(1)), i, 'on the sum')]
        # The lower word of Ra, taken as unsigned, added to the 64 bits of the uniform part; then the
        # upper word of Ra if it has one.
        pool = self.pool()
        new = [Item(mc.assemble('imad_wide__RsIR_RIR', Rd=pool, Ra=ops['Ra'], Sb=1, Rc=uniform, fmt='U32') | wait, i, 'address')]
        if d.c.name.endswith('Ra64'):
            new.append(Item(mc.assemble('iadd3_noimm__RRR_RRR', Rd=pool + 1, Ra=pool + 1, Rb=ops['Ra'] + 1, Rc=255) | wait, i, 'address'))
        new.append(self.late(target, self.put(target, out, 'Ra', pool), i, 'on the sum'))
        return new

    def read_announced(self, c, word, i):
        """`word` (of sm_70 class `c`) announcing on a scoreboard that it has read its operands: as
        it does already, or on the last scoreboard."""
        for kind, fields, data in self.mc.d70.plan(c):
            if kind == 'function' and data[1] == ['src_rel_sb']:
                if fields[0].get(word) != 7:
                    return word
                # Not the scoreboard on which it announces its result, if it has one.
                return fields[0].put(word, 5 if (word >> WRITE_BARRIER) & 7 != 5 else 4)
        raise Refused('%s at 0x%x cannot say when it has read its operands' % (c.name, 16 * i))

    def rule(self, d, to, i):
        mc = self.mc
        rule = mc.rule(d.c.name, to)
        if rule is None:
            raise Refused('no rule from %s to %s' % (d.c.name, to))
        out = rule.apply(d.word, self.R, self.P)
        if out is None:
            raise Refused('%s at 0x%x has a value that sm_70 does not have' % (d.c.name, 16 * i))
        return out

    def put(self, c, word, operand, value):
        """`word`, of class `c`, with its operand `operand` set to `value`."""
        dec = self.mc.d75 if c is self.mc.md75.by_name.get(c.name) else self.mc.d70
        for kind, fields, data in dec.plan(c):
            if kind == 'operand' and data == operand:
                return fields[0].put(word, value)
        raise Refused('%s has no operand %s' % (c.name, operand))

    # ---- relocations ---------------------------------------------------------------------------------------
    def follow(self, relocations):
        """Where each relocation on the code, (offset, kind), applies once the function is
        translated: on its instruction, which is there as it was, wherever it is now.

        One rewritten instruction also keeps its relocation: a number moved into a uniform
        register (`UMOV UR4, 32@lo(table)`), which becomes the same number moved into an ordinary
        register, with the number in the same 32 bits of the word, for the two kinds of
        relocation that write half of an address there. Any other instruction that the
        translation rewrites makes the function one it refuses: what is written at a place the
        instruction no longer has would be written into another one. (Until 2026-11-17 the
        relocations stayed where they were; PyTorch has none on code, cuFFT has them in 145
        kernels.)"""
        became = collections.defaultdict(list)
        for k, item in enumerate(self.items):
            if item.what not in AROUND:
                became[item.origin].append(k)
        moved = []
        for offset, kind in relocations:
            origin = offset // 16
            if origin >= len(self.src):
                raise Refused('a relocation outside its code')
            kept = False
            if len(became[origin]) == 1:
                item = self.items[became[origin][0]]
                as_it_was = not item.changed and item.fix is None and item.target is None
                half_of_an_address = (kind in HALVES and item.what == 'ordinary registers' and self.src[origin].c.name == 'umov__UI'
                                      and item.d.c.name == 'mov__RuIR_RI' and (item.word >> 32) & 0xffffffff == (self.words[origin] >> 32) & 0xffffffff)
                kept = as_it_was or half_of_an_address
            if not kept:
                raise Refused('a relocation on an instruction that the translation rewrites')
            moved.append(16 * became[origin][0] + offset % 16)
        return moved

    def without_batch(self, name, word):
        """`word`, a load the translation changed, with the batch of its original scheduling
        table cleared: what the translation changed names no batch (its schedule is the
        translation's). The wait hint of the table (its low five bits) stays."""
        cls = self.mc.md70.by_name[name]
        for kind, fields, data in self.mc.d70.plan(cls):
            if kind == 'table' and data[0] == 'TABLES_opex_0':
                opex = fields[0].get(word)
                if opex >> 5:
                    word = fields[0].put(word, opex & 31)
                break
        return word

    # ---- return addresses ----------------------------------------------------------------------------------
    def callee_returns_through_lepc(self, flow, target, register):
        """Whether every RET the callee can reach is RET.ABS on `register`, adding 0x20 (the
        call is two instructions after the LEPC), and nothing of the callee writes `register`.
        A callee that calls others, exits, or has an unreachable RET is not shown to return
        through it."""
        src = self.src
        if not 0 <= target < len(src):
            return False
        seen, work, rets = set(), [target], 0
        while work:
            i = work.pop()
            if i in seen:
                continue
            seen.add(i)
            d = src[i]
            if d.c.name.startswith('call') or d.c.name == 'exit_':
                return False
            if d.mnemonic == 'RET':
                if d.c.name not in ('ret__ABS', 'ret__ABS_UR') or d.ops.get('Ra') != register or d.ops.get('Ra_offset') != 0x20:
                    return False
                rets += 1
                if pairing.unguarded(d):
                    continue
                # A guarded return is one way out; the other falls through, and must be a way
                # back through the pair too.
            # LEPC Rd writes the pair Rd:Rd+1, and RET.ABS Ra reads the pair: a write of either
            # half between them makes the return address wrong.
            if any(l == 'R' and r != 255 and (r == register or r == register + 1) for l, r in d.writes):
                return False
            for t in flow.succ[i]:
                if 0 <= t < len(src):
                    work.append(t)
        return rets > 0

    def check_lepc_pairs(self):
        """The scheduling may put fillers between items: the LEPC and the call it gives the
        return address of must still be next to each other, or the callee returns into the
        filler."""
        for lepc_old, call_old in self.lepc_pairs:
            lepc = next((i for i, it in enumerate(self.items)
                         if 16 * it.origin == lepc_old and it.d is not None and it.d.mnemonic == 'LEPC'), None)
            call = next((i for i, it in enumerate(self.items)
                         if 16 * it.origin == call_old and it.d is not None and it.d.c.name.startswith('call_rel')), None)
            if lepc is None or call is None or call != lepc + 1:
                raise Refused('the LEPC at 0x%x and the call it gives the return address of are not adjacent after scheduling' % lepc_old)

    def return_addresses(self):
        """A callee is told where to return by a number: the offset of the instruction after the call,
        put in a register by a MOV before the call. When the code moves, so must that number. The MOV
        is found by what it does, and a function in which it is not found exactly is refused."""
        self.lepc_pairs = []
        mc, items = self.mc, self.items
        def immediate(item):
            d = item.d
            if d.c.name == 'mov__RuIR_RI' or (d.mnemonic == 'MOV' and 'Sb' in d.ops and 'Rb' not in d.ops and 'Sb_bank' not in d.ops):
                return d.ops['Sb']
            return None
        movs = collections.defaultdict(list)
        for k, item in enumerate(items):
            v = immediate(item)
            if v is not None:
                movs[v].append(k)
        first_of = {}
        for k, item in enumerate(items):
            if item.first:
                first_of[16 * item.origin] = k
        # A LEPC immediately before a direct call is that call's return address: both instructions
        # move with the code, so the address the LEPC reads stays the instruction after the call.
        lepc_links = set()
        for k in range(1, len(items)):
            if not items[k].d.c.name.startswith('call_rel') or 16 * items[k].origin in self.targets:
                continue
            if items[k - 1].d.mnemonic == 'LEPC':
                lepc_links.add(k - 1)
        # The callee decides; the flow of the source is built only when there is a pair to
        # judge. Only an accepted pair exempts its LEPC from "without a call after it".
        flow = pairing.Flow(self) if lepc_links else None
        accepted_lepc, accepted_call = set(), set()
        if flow is not None and flow.known:
            for k in lepc_links:
                item = items[k + 1]
                if item.target is not None and self.callee_returns_through_lepc(
                        flow, item.target // 16, items[k].d.ops['Rd']):
                    accepted_lepc.add(k)
                    accepted_call.add(k + 1)
        for k, item in enumerate(items):
            name = item.d.c.name
            if name == 'brx_' and 16 * item.origin in self.tables:
                # Its table is constants of the function (bank 2), read into the register by the
                # LDC before it: where that reads, the places are to be moved with the code.
                register = item.d.ops['Ra']
                define = next((j for j in range(k - 1, -1, -1) if ('R', register) in items[j].d.writes), None)
                load = items[define].d if define is not None else None
                if (load is None or not load.c.name.startswith('ldc__') or load.ops.get('Sa_bank') != 2 or load.ops.get('sz') != 4
                        or any((items[j].first and 16 * items[j].origin in self.targets) or (items[j].d.kind == 'BRU' and items[j].d.mnemonic not in ('BSSY', 'BMOV'))
                               for j in range(define + 1, k))):
                    raise Refused('BRX at 0x%x: its table is not read by an LDC before it' % (16 * item.origin))
                self.relocate.append((load.ops['Ra_offset'], self.tables[16 * item.origin]))
            elif name.startswith('call_rel') or name.startswith('jmp') or item.d.mnemonic in ('BRX', 'JMX', 'JMP'):
                if not name.startswith('call_rel'):
                    raise Refused('%s at 0x%x: a jump that is not understood' % (item.d.mnemonic, 16 * item.origin))
                back = 16 * item.origin + 16
                # The callee decides: when every one of its RETs returns with RET.ABS on the
                # register of the LEPC immediately before the call, and nothing of it writes
                # that register, the pair is this call's return address and no MOV of the same
                # number is one (it may be data). Otherwise the call is an ordinary one:
                # exactly one MOV of the return address before it, reaching it in a straight
                # line, is fixed up, as without a LEPC; none or several are refused.
                pair = k in accepted_call
                if pair:
                    self.lepc_pairs.append((16 * items[k - 1].origin, 16 * item.origin))
                else:
                    found = [j for j in movs.get(back, [])]
                    if len(found) != 1 or found[0] > k:
                        raise Refused('the return address of the call at 0x%x is not given by exactly one MOV' % (16 * item.origin))
                    j = found[0]
                    register = items[j].d.ops['Rd']
                    for between in items[j + 1:k]:
                        if ('R', register) in between.d.writes or (between.first and 16 * between.origin in self.targets) or between.d.kind == 'BRU':
                            raise Refused('the return address of the call at 0x%x does not reach it in a straight line' % (16 * item.origin))
                    items[j].fix = back
            elif name.startswith('call_abs'):
                # Through a register: the address to return to is the program counter (LEPC) plus the
                # difference of two numbers, added by the IADD3 that reads the three.
                lepc = next((j for j in range(k - 1, max(k - 16, -1), -1) if items[j].d.mnemonic == 'LEPC'), None)
                if lepc is None:
                    raise Refused('a call through a register at 0x%x without LEPC before it' % (16 * item.origin))
                pc = items[lepc].d.ops['Rd']
                add = next((j for j in range(lepc + 1, k) if items[j].d.c.name == 'iadd3_noimm__RRR_RRR' and ('R', pc, 'data') in items[j].d.reads), None)
                if add is None:
                    raise Refused('the call at 0x%x: no addition on the program counter' % (16 * item.origin))
                ops = items[add].d.ops
                terms = [(ops[r], ops.get(r + '@negate', 0)) for r in ('Ra', 'Rb', 'Rc') if ops[r] != pc]
                if len(terms) != 2 or sorted(n for _, n in terms) != [0, 1]:
                    raise Refused('the call at 0x%x: the addition on the program counter has another shape' % (16 * item.origin))
                values = {}
                for register, negated in terms:
                    define = next((j for j in range(add - 1, -1, -1) if ('R', register) in items[j].d.writes), None)
                    if define is None or immediate(items[define]) is None or any(items[j].first and 16 * items[j].origin in self.targets for j in range(define + 1, k + 1)):
                        raise Refused('the call at 0x%x: a term of its return address is not a MOV before it' % (16 * item.origin))
                    values[negated] = define
                if any(items[j].first and 16 * items[j].origin in self.targets for j in range(lepc + 1, k + 1)):
                    raise Refused('the call at 0x%x: a branch arrives between LEPC and the call' % (16 * item.origin))
                back = 16 * items[lepc].origin + immediate(items[values[0]]) - immediate(items[values[1]])
                if back % 16 or not 0 <= back < 16 * len(self.src):
                    raise Refused('the call at 0x%x: its return address 0x%x is not an instruction' % (16 * item.origin, back))
                items[values[0]].fix, items[values[1]].fix = back, 16 * items[lepc].origin
            elif item.d.mnemonic == 'LEPC':
                if k not in accepted_lepc and not any(items[j].d.c.name.startswith('call_abs') for j in range(k + 1, min(len(items), k + 16))):
                    raise Refused('LEPC at 0x%x without a call after it' % (16 * item.origin))

    # ---- scheduling -----------------------------------------------------------------------------------------
    def schedule(self):
        """Raise stall counts where an instruction of the translation is read, or reads, closer than
        native sm_70 code allows: within straight-line code by the measured distance for the pair, and
        at the end of it by the longest distance any reader could need (for a writer that names no
        scoreboard, and for a result of double precision of code for sm_80, which is not read
        early even when waited for)."""
        mc, items = self.mc, self.items
        blocks, start = [], 0
        for k, item in enumerate(items):
            if k > start and item.first and 16 * item.origin in self.targets:
                blocks.append((start, k))
                start = k
            if item.d.kind == 'BRU':
                blocks.append((start, k + 1))
                start = k + 1
        if start < len(items):
            blocks.append((start, len(items)))
        inserted = []
        t, last, waited = 0, {}, set()
        for a, b in blocks:
            # What was written before this block, and when, is known where the code before it goes
            # on into it (past a label, a branch not taken, the mark of a region): an instruction
            # the translation changed or added here is as far from those writers as from the
            # ones of its own block. Where the code before does not go on (or goes through other
            # code first: a call), nothing is known, as for what jumps here.
            if not self.goes_on(a):
                t, last, waited = 0, {}, set()
            read = {}
            for k in range(a, b):
                item = items[k]
                need = 0
                made_up = item.more_reads + self.all_predicates(item)
                for letter, register, role in item.d.reads + made_up:
                    w = last.get((letter, register))
                    if w is None or not (w[1].changed or item.changed):
                        continue
                    if w[2] and (letter, register, role) not in made_up:
                        # (Written through the mask of an R2P of the program, and read by what
                        # the program's instruction read: as far as the compiler had them.)
                        continue
                    if (w[1].word >> WRITE_BARRIER) & 7 != 7 or w[1].covered:
                        # Announced on a scoreboard: the reader waits for it. What the program's
                        # instruction waited for, its translation waits for; a read that the
                        # translation made up (a predicate saved, written down, or all of them
                        # for R2P) has nothing of the program that waits for it: it does here,
                        # unless an instruction since has waited for that scoreboard (`waited`).
                        if (letter, register, role) in made_up and (w[1].word >> WRITE_BARRIER) & 7 != 7 and id(w[1]) not in waited:
                            if not item.changed:
                                raise Refused('the %s at 0x%x reads a predicate of the translation that a scoreboard announces' % (item.d.mnemonic, 16 * item.origin))
                            self.waits_for(item, (w[1].word >> WRITE_BARRIER) & 7)
                        # (A result of double precision is not read earlier than Volta's own
                        # code reads one, whatever is waited for: `Machine.waited`.)
                        floor = mc.waited(letter, role, w[1].d, item.d.kind)
                        if floor is not None:
                            need = max(need, w[0] + floor - t)
                        continue
                    need = max(need, w[0] + mc.least(letter, role, w[1].d, item.d.kind) - t)
                if item.changed and (item.d.kind not in ('FXU', 'IMAD', 'FMAI') or item.d.mnemonic == 'R2P'):
                    # A writer that is not plain arithmetic, after a reader of what it writes: the
                    # compiler's tables ask for up to 9 cycles between them (2 in most cases); the
                    # original had another kind of writer here, with its own distance. (Not a
                    # product after a product: they are done in order, and the compiler has the
                    # steps of one product two cycles apart, the third writing what the first read.)
                    # R2P is kept to this too, for another reason than the one first given for it
                    # (2026-11-18, the same day: a reader does NOT go by what an R2P issued after
                    # it gives back, at any distance): the instruction before a
                    # restore reads what stood in the predicate, so the restore is nine cycles
                    # into its block or more, and no writer of a predicate that ran before the
                    # block, on a path the schedule does not see, is close enough for the R2P to
                    # undo (`all_predicates`). What R2P writes is in a mask: `more_writes`.
                    for letter, register in item.d.writes + item.more_writes:
                        r = read.get((letter, register))
                        if r is not None and not (item.d.kind == 'HMMA' and r[1] == 'HMMA'):
                            need = max(need, r[0] + 9 - t)
                if need > 0:
                    if k == 0 or (k == a and not self.goes_on(a)):
                        raise Refused('scheduling: nothing before 0x%x to wait on' % (16 * item.origin))
                    t += self.delay(k - 1, need, inserted)
                # (As a reader for the nine cycles above: a P2R with the predicates of its mask;
                # not an R2P with those it only writes back.)
                for letter, register, role in item.d.reads + item.more_reads + (self.all_predicates(item) if item.d.mnemonic == 'P2R' else []):
                    read[(letter, register)] = (t, item.d.kind)
                # (What this instruction waits for is there when it is issued: the writers that
                # announced on those scoreboards are waited for.)
                boards = (item.word >> WAIT) & 63
                if boards:
                    waited |= {id(w) for when, w, through in last.values() if boards >> ((w.word >> WRITE_BARRIER) & 7) & 1}
                for letter, register in item.d.writes + item.more_writes:
                    last[(letter, register)] = (t, item, False)
                for letter, register in self.masked(item):
                    last[(letter, register)] = (t, item, True)
                t += stall_of(item.word)
            tail = b - 1 if items[b - 1].d.kind != 'BRU' or b - a == 1 else b - 2
            need = 0
            for (letter, register), (when, item, through) in last.items():
                if item.changed and (item.word >> WRITE_BARRIER) & 7 == 7 and not item.covered and item is not items[b - 1]:
                    need = max(need, when + mc.longest(letter, item.d) - t)
                # (A result of double precision of code for sm_80, scoreboard or not: its reader
                # may be the first instruction where a branch goes, which waits and is still too
                # early. Code for sm_75 keeps Volta's distances along its branches too: none
                # shorter in the 475 cubins of PyTorch, and the translation shortens none.)
                floor = mc.waited_longest(letter, item.d)
                if item.anew and floor is not None and item is not items[b - 1]:
                    need = max(need, when + floor - t)
            if items[b - 1].changed and items[b - 1].d.writes + items[b - 1].more_writes and (items[b - 1].word >> WRITE_BARRIER) & 7 == 7:
                tail = b - 1
                need = max(need, max(mc.longest(l, items[b - 1].d) for l, r in items[b - 1].d.writes + items[b - 1].more_writes) - stall_of(items[b - 1].word))
            if items[b - 1].anew and items[b - 1].d.writes and mc.waited_longest('R', items[b - 1].d) is not None:
                tail = b - 1
                need = max(need, max(mc.waited_longest(l, items[b - 1].d) for l, r in items[b - 1].d.writes) - stall_of(items[b - 1].word))
            if need > 0:
                t += self.delay(tail, need, inserted)
        for k, word in sorted(inserted, reverse=True):
            filler = Item(word, items[k].origin, 'wait', changed=True)
            filler.d = mc.decode(70, word)
            items.insert(k + 1, filler)
        # A wait for a scoreboard comes two cycles or more after the instruction that names it, for
        # its result or for having read: native code of both architectures never has less (some
        # 700,000 such pairs in PyTorch's, none closer). The translation adds waits and names.
        for k in range(1, len(items)):
            mask = (items[k].word >> WAIT) & 63
            distance = 0
            for j in range(k - 1, -1, -1):
                if not mask:
                    break
                distance += stall_of(items[j].word)
                if distance >= 2:
                    break
                named = {(items[j].word >> WRITE_BARRIER) & 7, (items[j].word >> READ_BARRIER) & 7} - {7}
                if (items[j].changed or items[k].changed or items[j].says) and any(mask >> s & 1 for s in named):
                    items[k - 1].word = with_stall(items[k - 1].word, stall_of(items[k - 1].word) + 2 - distance, marked(items[k - 1].d))
                    break

    def goes_on(self, a):
        """Whether the item before item `a`, the first of a block of the schedule, goes on into it
        with nothing else run between: an instruction that is not a branch, a branch that may not
        be taken, the mark of a region (BSSY, BMOV). Not a call (other code runs first), nor
        anything that leaves."""
        if a == 0:
            return False
        d = self.items[a - 1].d
        if d.kind != 'BRU' or d.mnemonic in ('BSSY', 'BMOV'):
            return True
        ops = d.ops
        always = (ops.get('Pg', 7) == 7 and not ops.get('Pg@not') and ops.get('Pp', 7) == 7 and not ops.get('Pp@not')
                  and not ops.get('cond'))
        return d.mnemonic == 'BRA' and not always

    def waits_for(self, item, board):
        """Make `item` wait for scoreboard `board`."""
        item.word |= 1 << (WAIT + board)

    def all_predicates(self, item):
        """What a P2R or an R2P reads of the predicates, which its operands do not say (they are a
        mask, a number).

        A P2R reads those of its mask.

        An R2P reads all seven and writes them all back, those of its mask from its register and
        the others as it read them; so a predicate outside its mask that an instruction wrote
        less than 4 cycles before it (5 for another R2P) gets its former value back. Measured on
        a V100, 2026-11-18 (ISETP, PLOP3 or the carries of IADD3 1 to 3 cycles
        before an R2P of another predicate are undone; two R2P of one predicate each, 1 to 4
        cycles apart, lose the first). Native sm_70 code has no writer of any predicate closer
        than 4 cycles before an R2P (8,101 of them in PyTorch's and cuBLAS's). The translation
        had restores one cycle apart (two predicates saved around one instruction) and so lost
        the first. (Not the predicates of its mask, which get the register's bits whatever they
        held; a mask that is not a number: any of them may be left as it was.)"""
        d = item.d
        named = {(letter, register) for letter, register, role in d.reads + item.more_reads}
        if d.mnemonic == 'P2R':
            mask = d.ops['Sb'] & 0x7f if 'Sb' in d.ops else 0x7f
            return [('P', p, 'data') for p in range(7) if mask >> p & 1 and ('P', p) not in named]
        if d.mnemonic == 'R2P':
            mask = d.ops['Sb'] & 0x7f if d.c.name == 'r2p__RIR' else 0
            return [('P', p, 'data') for p in range(7) if not mask >> p & 1 and ('P', p) not in named]
        return []

    def masked(self, item):
        """What an R2P writes of the predicates, which its operands do not say: those of its mask
        (any of them when the mask is not a number). The program's own R2P is a writer like any
        other for what the translation reads after it: a predicate saved (P2R) one to four cycles
        after the program's R2P gave it a value was saved as it was before (cuDNN's kernels had
        the case, found by validate.py on the held-out corpus the day R2P was measured)."""
        d = item.d
        if d.mnemonic != 'R2P':
            return []
        mask = d.ops['Sb'] & 0x7f if d.c.name == 'r2p__RIR' else 0x7f
        named = set(d.writes + item.more_writes)
        return [('P', p) for p in range(7) if mask >> p & 1 and ('P', p) not in named]

    def delay(self, k, cycles, inserted):
        """Make what follows item `k` start `cycles` later: a longer stall on it, and instructions
        that do nothing but wait after it when a stall cannot hold it all."""
        item = self.items[k]
        room = 15 - stall_of(item.word)
        take = min(room, cycles)
        item.word = with_stall(item.word, stall_of(item.word) + take, marked(item.d))
        left = cycles - take
        while left > 0:
            step = min(15, left)
            inserted.append((k, with_stall(NOP, step)))
            left -= step
        return cycles

    # ---- the result ------------------------------------------------------------------------------------------
    def finish(self):
        mc, items = self.mc, self.items
        new_of = {}
        for k, item in enumerate(items):
            if item.first:
                new_of[16 * item.origin] = 16 * k
        new_of[16 * len(self.src)] = 16 * len(items)
        for k, item in enumerate(items):
            if item.target is not None:
                rel = relative(mc, 70, item.d)
                if rel is None:
                    raise Refused('the translation of the branch at 0x%x has nowhere to say where it goes' % (16 * item.origin))
                field, scale = rel
                item.word = field.put(item.word, ((new_of[item.target] - 16 * k - 16) // scale) & ((1 << len(field)) - 1))
            if item.fix is not None:
                item.word = self.put(item.d.c, item.word, 'Sb', new_of[item.fix])
            if mc.decode(70, item.word) is None:
                raise Refused('after its operands were set, the instruction for 0x%x is not an sm_70 instruction' % (16 * item.origin))
        code = b''.join(item.word.to_bytes(16, 'little') for item in items)
        # The register count of a function is two more than the registers it uses (native code of both
        # architectures has it so, and Volta faults on a register in the last two).
        return code, new_of, self.next_register + 2


def attributes(data, new_of, name):
    """The attributes of a function with their code offsets moved."""
    out, pos = bytearray(data), 0
    while pos + 4 <= len(data):
        fmt, attr, val = data[pos], data[pos + 1], struct.unpack_from('<H', data, pos + 2)[0]
        size = 4 + val if fmt == 4 else 4
        if fmt == 4:
            if attr in OFFSET_LISTS:
                places = range(pos + 4, pos + size, 4)
            elif attr in OFFSET_PAIRS:
                places = range(pos + 4 + 4 * OFFSET_PAIRS[attr], pos + size, 8)
            elif attr in PLAIN:
                places = ()
            elif attr == 0x34:
                # Jump tables (`tables_of`): the branch and the places it leads to.
                places, at = [], pos + 4
                while at + 12 <= pos + size:
                    count = struct.unpack_from('<I', data, at + 8)[0] if at + 12 <= len(data) else 0
                    if at + 12 + 4 * count > pos + size:
                        raise Refused('a record of jump tables that is not understood')
                    places += [at] + list(range(at + 12, at + 12 + 4 * count, 4))
                    at += 12 + 4 * count
            else:
                raise Refused('attribute 0x%02x, which is not known' % attr)
            for at in places:
                if at + 4 > len(data):
                    continue                    # an attribute cut short
                old = struct.unpack_from('<I', data, at)[0]
                if old not in new_of:
                    raise Refused('attribute 0x%02x points at 0x%x, which is not an instruction' % (attr, old))
                struct.pack_into('<I', out, at, new_of[old])
        pos += size
    return bytes(out)


def without(data, drop):
    """A set of attributes without those of the kinds in `drop`."""
    out, pos = bytearray(), 0
    while pos + 4 <= len(data):
        size = 4 + struct.unpack_from('<H', data, pos + 2)[0] if data[pos] == 4 else 4
        if data[pos + 1] not in drop:
            out += data[pos:pos + size]
        pos += size
    return bytes(out)


def write_lives(path, lives):
    """The lives of uniform predicates of a translation, for validate.py: next to the map. And,
    with borrowed registers (VF_BORROW), what was borrowed where: lines `home` and `pool`,
    and at level 3 `slot` and `frame`."""
    with open(path, 'w') as f:
        for name, q, p, saved, offsets in lives:
            # (`held`: a life kept in a register between the instructions that name the uniform
            # predicate; `p` is then that register, and each of those instructions has a life
            # of its own, of one instruction, for the ordinary predicate it uses.)
            how = saved if isinstance(saved, str) else 'saved' if saved else 'free'
            f.write('%s\t%d\t%d\t%s\t%s\n' % (name, q, p, how, ','.join('%x' % at for at in offsets)))


def relocations_of(e, symbols, code):
    """The relocations of a cubin: what the driver, or a linker, writes into the instructions and
    the data they name once it knows the addresses. Returned: for each section of code (`code`
    has their indices), the ones on it, as (section of relocations, where in it, offset in the
    code, kind); the sections of code that hold one which names a place in code; the sections of
    code that one names. (Those of the debug sections are for a debugger.)"""
    relocations, holders, named = collections.defaultdict(list), set(), set()
    for s in e.sections:
        if s.type not in (elf.SHT_REL, elf.SHT_RELA) or s.info >= len(e.sections):
            continue
        if e.sections[s.info].name.startswith(('.debug', '.nv_debug')):
            continue
        step = 24 if s.type == elf.SHT_RELA else 16
        for at in range(0, len(s.data) - step + 1, step):
            offset, info = struct.unpack_from('<QQ', s.data, at)
            place = symbols[info >> 32][3] if info >> 32 < len(symbols) else None
            if s.info in code:
                relocations[s.info].append((s, at, offset, info & 0xffffffff))
                if place in code:
                    holders.add(s.info)
            if place in code:
                named.add(place)
    return relocations, holders, named


def assembler_helper(e, symbols, index):
    # CUDA 12 ptxas places certain weak helpers before the first global symbol.
    # The kernel owns the entire helper body; see cubin_debug_ptx in vf-fatbin.
    name, info, other, sec, start, size, _ = symbols[index]
    symtab = next((s for s in e.sections if s.type == elf.SHT_SYMTAB), None)
    if (not symtab or not index < symtab.info <= len(symbols) or info != 0x22 or other or
            not 0 < sec < len(e.sections) or
            not re.fullmatch(r'\$__internal_[0-9]+_\$__cuda_.+', name, flags=re.DOTALL)):
        return False
    section = e.sections[sec]
    owner_index = section.info & 0xffffff
    if section.type != 1 or not section.flags & 4 or owner_index >= len(symbols):
        return False
    _, owner_info, owner_other, owner_sec, base, length, _ = symbols[owner_index]
    return (owner_info & 15 == 2 and owner_other & 0x10 and owner_sec == sec and size > 0 and
            base <= start and start + size <= base + length)


def translate(data, mc=None, listing=None, lives=None):
    """The sm_70 cubin for the sm_75 cubin `data`, and a report: {function: None or why it was refused}.
    `listing`, a list, gets one row for each instruction of the result (function, new offset, old
    offset, what was done); `lives`, a list, one for each life of a uniform predicate (function,
    uniform predicate, the ordinary one that stands for it, whether that one is saved around the
    life, the old offsets of the life): what validate.py checks."""
    mc = mc or MC.Machine()
    e = elf.Elf(data)
    arch = e.header[0x31 if e.header[8] >= 8 else 0x30]
    if arch not in (75, 80):
        raise Refused(f'unsupported machine-code target sm_{arch}')
    if arch == 80 and (tuple(e.header[7:9]) not in ((0x33, 7), (0x41, 8)) or e.header[16:20] != bytes((2,0,190,0))):
        raise Refused('sm_80 translation requires an executable CUDA ELF with a known ABI')
    symbols = e.symbols()
    texts = [s for s in e.sections if s.name.startswith('.text.') and s.type == 1]
    assert len({s.name for s in texts}) == len(texts), 'two sections of code with the same name'
    index_of = {e.index(s): s for s in texts}
    # Functions that call one another share their registers: one group, one place for the uniform ones.
    group = {e.index(s): e.index(s) for s in texts}
    def find(x):
        while group[x] != x:
            x = group[x]
        return x
    by_symbol = {i: sym[3] for i, sym in enumerate(symbols) if sym[3] in group}
    # The sections that are kernels: a function symbol with the mark of an entry point.
    entries = {sym[3] for sym in symbols if sym[1] & 15 == 2 and sym[2] & 0x10}
    relocations, holders, named = relocations_of(e, symbols, group)
    graph = e.section('.nv.callgraph')
    edges = []
    if graph is not None:
        for at in range(0, len(graph.data) - 7, 8):
            a, b = struct.unpack_from('<Ii', graph.data, at)
            if b >= 0 and a in by_symbol and b in by_symbol:
                edges.append((by_symbol[a], by_symbol[b]))
    for s in e.sections:
        if s.name.startswith('.nv.info.'):
            text = e.section('.text.' + s.name[len('.nv.info.'):])
            pos = 0
            while text is not None and pos + 4 <= len(s.data):
                fmt, attr, val = s.data[pos], s.data[pos + 1], struct.unpack_from('<H', s.data, pos + 2)[0]
                if fmt == 4 and attr == 0x0f:
                    for at in range(pos + 4, min(pos + 4 + val, len(s.data)) - 3, 4):
                        callee = struct.unpack_from('<I', s.data, at)[0]
                        if callee in by_symbol:
                            edges.append((e.index(text), by_symbol[callee]))
                pos += 4 + val if fmt == 4 else 4
    for a, b in edges:
        group[find(a)] = find(b)
    report, functions = {}, {}
    for s in texts:
        name = s.name[len('.text.'):]
        try:
            info = e.section('.nv.info.' + name)
            sec = e.index(s)
            code = s.data
            if arch == 80:
                secondary = [(idx, y) for idx, y in enumerate(symbols) if y[3] == sec and y[1] & 15 == 2 and y[4]]
                helper = None
                if secondary:
                    index, (_, info_byte, other, _, at, size, _) = secondary[0]
                    if len(secondary) != 1 or (info_byte >> 4 != 0 and not assembler_helper(e, symbols, index)) or other & 0x10 or not size or at % 16 or at + size != len(code):
                        raise Refused('unsupported secondary entry in sm_80 code')
                    helper = at // 16
                if relocations.get(sec):
                    raise Refused('a relocation in sm_80 code')
                code, operations = ampere.normalize(code, mc, relative, signed, Refused, helper)
            functions[sec] = Function(mc, name, code, s.info >> 24, tables_of(info.data) if info is not None else {})
            functions[sec].ampere = arch == 80
            if arch == 80:
                functions[sec].ampere_ops = operations
                f = functions[sec]
                for i, op in operations.items():
                    if op == 'pack':
                        f.needs_pool |= f.src[i].ops['Rc'] != 255
                    elif op in ('bf16add', 'bf16pack'):
                        f.scratch_need = max(f.scratch_need, 3)
                    else:
                        f.scratch_need = max(f.scratch_need, 2 if op[0] == "increment" else 6)
            report[name] = None
        except Refused as why:
            report[name] = str(why)
    for symbol in symbols:
        # A function of its own within the section of another (the helpers that the code of the
        # section calls are weak symbols): it is entered there, not at the start of the section.
        if symbol[1] & 15 == 2 and symbol[1] >> 4 == 1 and symbol[4] != 0 and symbol[3] in functions:
            functions[symbol[3]].entered_inside = True
    for i in sorted(holders | named):
        # A relocation that names a place in code: the place moves with the translation, and
        # nothing here carries that over. The function it names and the function it is in.
        if i in functions:
            report[functions.pop(i).name] = 'a relocation in its code names a place in code' if i in holders else 'a relocation names a place in its code'
    members = collections.defaultdict(list)
    for i in group:
        members[find(i)].append(i)
    results = {}
    for root, indices in members.items():
        if any(i not in functions for i in indices):
            for i in indices:
                if i in functions:
                    report[functions[i].name] = 'a function it calls, or that calls it, is refused'
            continue
        fs = [functions[i] for i in indices]
        try:
            if len(fs) > 1 and any(f.uniform_predicates for f in fs):
                raise Refused('uniform predicates in functions that call one another')
            declared = max(f.registers for f in fs)
            for f in fs:
                if f.highest >= declared:
                    raise Refused('register R%d is used and the function declares %d' % (f.highest, declared))
            for f in fs:
                f.stand_ins()
            # The registers of the translation start after the last one any of them uses.
            def attempt(pairs, companions=False, borrow=0, top=None):
                # (`borrow`: 2, the uniform registers that find a home in the function's own
                # registers have it and the pool is taken there at each instruction; 1, the homes
                # only, the pool above; 0, everything above, as it was.
                # 3: level 3, everything at the instruction: unhomed uniform registers in local
                # memory, pool and scratch and lane taken at each instruction.)
                for f in fs:
                    f.plan_pairs(pairs, companions)
                    f.borrow = None
                if borrow:
                    # (A kernel alone: nothing else shares its registers. `Borrowed`.)
                    # (`VF_BORROW=3`, for tests: no homes, every uniform register in memory.)
                    fs[0].borrow = Borrowed(fs[0], all_in_memory=(borrow == 3 and BORROW in (3, 8)), top=top)
                if borrow == 3:
                    # Level 3, everything at the instruction: the uniform
                    # registers that find no home live in the thread's local memory, in a frame
                    # added on top of the program's; the pool, the scratch registers and the
                    # lane of a matrix load, and a register for each uniform register of the
                    # frame that an instruction names, are taken at that instruction among the
                    # registers that hold nothing there.
                    # At VF_BORROW >= 7 a function with a matrix product is translated at
                    # level 3: the lane is read at each product, the four values computed
                    # there, B gathered every time, and every register taken at the instruction
                    # (docs/translator.md, "Registers"). At VF_BORROW < 7: refused as before.
                    f = fs[0]
                    if f.has_product and BORROW < 7:
                        raise Refused('a matrix product in a function at level 3')
                    b = f.borrow
                    b.everything = True
                    b.assign_slots(sorted(f.uniform), f.pairs)
                    # (`VF_BORROW=5`: room is made where there is none, and the frame is
                    # addressed from the stack pointer; in a function that never moves it.)
                    b.room = b.from_r1 = BORROW >= 5 and b.r1_fixed
                    # A uniform predicate held in a register between the instructions that name
                    # it: one register that holds nothing through all its lives.
                    spans = collections.defaultdict(set)
                    for q, part in f.held_lives:
                        spans[q].update(part)
                    held_at = {}
                    for q, span in sorted(spans.items()):
                        r = b.kept(sorted(span))
                        if r is not None:
                            held_at[q] = r
                        elif b.from_r1:
                            b.predicate_slots[q] = b.slot_count()
                            b.frame_bytes = (4 * b.slot_count() + 15) & ~15
                        else:
                            raise Refused('no register holds nothing through the lives of uniform predicate UP%d' % q)
                    # Above the registers that may be taken: those the predicates are saved in, only.
                    where = placement(b.top + 1, [], set(), False, f.save_slots, False, 0, False, {}, [], {}, [])
                    out = {indices[0]: f.translate(dict(b.homes), None, where[2], None, None, None, None, where[7], {}, {}, {}, held_at)}
                    if os.environ.get('VF_BORROW_SAY'):
                        print('  registers\t%s\t%d\t%d\tlevel 3\tuniform %d\thomed %d\tin memory %d\tframe %d\tsaves %d\theld %d\tproducts %d\tgave up: %s'
                              % (f.name, f.registers, max(r[2] for r in out.values()), len(f.uniform), len(b.homes),
                                 len(b.slots), b.frame_bytes, f.save_slots, len(held_at), f.product_count, '; '.join(gave_up) or '-'), file=sys.stderr)
                    return out
                # The registers of the rows: one set for each destination of whole-row loads, or
                # one for several whose rows are never wanted at the same time (a function alone
                # in its group: `pairing.sharing`).
                rows, shares = {}, {}
                for f in fs:
                    for rd, count in (f.q.rows() if f.q else {}).items():
                        to = f.q.share.get(rd, rd) if len(fs) == 1 else rd
                        shares[rd] = to
                        rows[to] = max(rows.get(to, 0), count)
                row_kept = [name for name in ROW_KEPT if any(f.q and name in f.q.kept() for f in fs)]
                beside = {}
                for f in fs:
                    for root, count in (f.q.companion_registers() if f.q else {}).items():
                        beside[root] = max(beside.get(root, 0), count)
                homes = fs[0].borrow.homes if borrow else {}
                where = placement(max(f.highest for f in fs) + 1, sorted({u for f in fs for u in f.uniform} - set(homes)),
                                  {u for f in fs for u in f.pairs} - set(homes), any(f.needs_pool for f in fs) and borrow != 2,
                                  max(f.save_slots for f in fs),
                                  any(f.needs_lane for f in fs), max(f.scratch_need for f in fs), any(f.has_product for f in fs),
                                  rows, row_kept, beside, sorted({q for f in fs for q in f.held_predicates}))
                where = ({**where[0], **homes},) + where[1:8] + ({rd: where[8][to] for rd, to in shares.items()},) + where[9:]
                out = {i: f.translate(*where) for i, f in zip(indices, fs)}
                if os.environ.get('VF_BORROW_SAY') and len(fs) == 1:
                    # (One line a function, for counts: what it had, what it has, and where the
                    # difference went.)
                    f = fs[0]
                    print('  registers\t%s\t%d\t%d\tlevel %d\tuniform %d\thomed %d\tpool %d\tsaves %d\theld %d\tproducts %d\tgave up: %s'
                          % (f.name, f.registers, max(r[2] for r in out.values()), borrow, len(f.uniform), len(homes), 2 * bool(f.needs_pool and borrow != 2),
                             f.save_slots, len(f.held_predicates), f.product_count, '; '.join(gave_up) or '-'), file=sys.stderr)
                return out
            gave_up = []
            alone = len(fs) == 1 and indices[0] in entries
            def usual():
                try:
                    if BORROW and alone:
                        try:
                            return attempt(True, True, 2)
                        except Refused as why:
                            gave_up.append(str(why))
                            if os.environ.get('VF_PAIRS_SAY'):
                                print('  without a borrowed pool, %s: %s' % (fs[0].name[:50], why), file=sys.stderr)
                            try:
                                return attempt(True, True, 1)
                            except Refused as why:
                                gave_up.append(str(why))
                                if os.environ.get('VF_PAIRS_SAY'):
                                    print('  without borrowed registers, %s: %s' % (fs[0].name[:50], why), file=sys.stderr)
                                return attempt(True, True)
                    return attempt(True, True)
                except Refused as why:
                    # Neither companions nor pairs on whole rows make a function refused that is
                    # translated without them: without the companions first, then without the pairs.
                    if not any(f.q for f in fs):
                        raise
                    try:
                        if not any(f.q.roots for f in fs if f.q):
                            raise
                        if os.environ.get('VF_PAIRS_SAY'):
                            print('  without companions, %s: %s' % (fs[0].name[:50], why), file=sys.stderr)
                        return attempt(True, False)
                    except Refused as why:
                        if os.environ.get('VF_PAIRS_SAY'):
                            print('  without pairs, %s: %s' % (fs[0].name[:50], why), file=sys.stderr)
                        return attempt(False)
            def plain():
                # (As with the switch off: everything above the program's registers.)
                try:
                    return attempt(True, True)
                except Refused as why:
                    if not any(f.q for f in fs):
                        raise
                    try:
                        if not any(f.q.roots for f in fs if f.q):
                            raise
                        return attempt(True, False)
                    except Refused:
                        return attempt(False)
            if BORROW == 8 and alone:
                # For tests: level 3 first for EVERY kernel, products included (as 3 is
                # without products). With room made: a small function that has not enough
                # registers for the product's scratch at once gets them above its own (top
                # extended by 16 for the product, as it would at level 0). Falls back to
                # the usual chain when level 3 is refused.
                f = fs[0]
                extra = 16 if f.has_product else 0
                try:
                    done = attempt(False, False, 3, f.highest + extra if extra else None)
                except Refused as why:
                    gave_up.append(str(why))
                    done = usual()
            elif BORROW in (7, 9) and alone:
                # What 6 does, and a function with products may be translated at level 3.
                # The chain: plain; levels 1, 2; level 3 with room and with products.
                way = plain
                try:
                    done = way()
                except Refused as why:
                    if str(why) != 'more registers than there are':
                        raise
                    gave_up.append(str(why))
                    done = None
                    for way in (lambda: attempt(True, True, 1), lambda: attempt(True, True, 2), lambda: attempt(False, False, 3)):
                        try:
                            done = way()
                            break
                        except Refused as also:
                            gave_up.append(str(also))
                    if done is None:
                        if os.environ.get('VF_BORROW_SAY'):
                            print('  refused\t%s\t%s\tand with borrowed registers: %s' % (fs[0].name, why, '; '.join(gave_up[1:])), file=sys.stderr)
                        raise why
                if BORROW == 9:
                    f = fs[0]
                    info = e.section('.nv.info.' + f.name)
                    limit = graph_register_budget(f.registers, info.data if info is not None else b'')
                    if done[indices[0]][2] > limit and not relocations.get(indices[0]) and not f.tables:
                        top = max(0, limit - 3 - f.save_slots)
                        for bounded in (lambda: attempt(True, True, 1), lambda: attempt(True, True, 2),
                                        lambda: attempt(False, False, 3, top)):
                            try:
                                candidate = bounded()
                            except Refused:
                                continue
                            if candidate[indices[0]][2] <= limit:
                                # Preserve all instruction boundaries used by attributes and symbols.
                                done = candidate if candidate[indices[0]][1].keys() == done[indices[0]][1].keys() else way()
                                break
                        else:
                            # Restore the instruction map, frame and relocations of the normal translation.
                            done = way()
            elif BORROW == 6 and alone:
                # What the layer does by itself. Wherever the translation of the switch off
                # exists it is the one taken, byte for byte: nothing that runs today changes,
                # in what it computes or in its speed (measured: borrowed registers cost some
                # operations of PyTorch up to a fifth of their time). A kernel
                # that is refused for want of registers gets, in this order: homes (level 1),
                # the pool at the instruction too (level 2), then everything at the instruction,
                # with room made where there is none (level 3). And when none translates, it
                # is refused for the reason it was refused before.
                try:
                    done = plain()
                except Refused as why:
                    if str(why) != 'more registers than there are':
                        raise
                    gave_up.append(str(why))
                    done = None
                    for way in (lambda: attempt(True, True, 1), lambda: attempt(True, True, 2), lambda: attempt(False, False, 3)):
                        try:
                            done = way()
                            break
                        except Refused as also:
                            gave_up.append(str(also))
                    if done is None:
                        if os.environ.get('VF_BORROW_SAY'):
                            print('  refused\t%s\t%s\tand with borrowed registers: %s' % (fs[0].name, why, '; '.join(gave_up[1:])), file=sys.stderr)
                        raise why
            elif BORROW >= 4 and alone:
                # By the budget: a V100 gives a thread its registers by eights, and has 65,536 of
                # them a multiprocessor for at most 2,048 threads: a kernel loses no thread as
                # long as its count stays in the eight it was in, or under 32. The levels in the
                # order of the speed of what they make (0: everything above, nothing to wait
                # for; 1: homes; 2: and the pool at the instruction; 3: everything at the
                # instruction, with the registers left under the budget, and memory): the first
                # that stays within the budget, or else the one that takes the fewest.
                f = fs[0]
                budget = min(max((f.registers + 7) // 8 * 8, 32), 255)
                ways = [(0, plain), (1, lambda: attempt(True, True, 1)), (2, lambda: attempt(True, True, 2)),
                        (3, lambda: attempt(False, False, 3, budget - 3 - f.save_slots))]
                best, first_why = None, None
                for level, way in ways:
                    try:
                        done = way()
                    except Refused as why:
                        gave_up.append(str(why))
                        first_why = first_why or why
                        continue
                    count = max(r[2] for r in done.values())
                    if count <= budget:
                        best = (count, level, way, True)
                        break
                    if best is None or count < best[0]:
                        best = (count, level, way, False)
                if best is None:
                    raise first_why
                if not best[3]:
                    done = best[2]()            # (again: the function is left as its last attempt made it)
                if os.environ.get('VF_BORROW_SAY'):
                    print('  budget\t%s\t%d\t%d\t%d\tlevel %d\t%s' % (f.name, f.registers, best[0], budget, best[1], 'within' if best[3] else 'over'), file=sys.stderr)
            elif BORROW == 3 and alone:
                # (For tests: everything at the instruction first, every uniform register in memory.)
                try:
                    done = attempt(False, False, 3)
                except Refused as why:
                    gave_up.append(str(why))
                    done = usual()
            elif BORROW == 2 and alone:
                # Everything at the instruction for a kernel that is refused otherwise; and when
                # that is refused too, for the reason it was refused before.
                try:
                    done = usual()
                except Refused as why:
                    gave_up.append(str(why))
                    try:
                        done = attempt(False, False, 3)
                    except Refused as also:
                        if os.environ.get('VF_BORROW_SAY'):
                            print('  refused\t%s\t%s\tat level 3: %s' % (fs[0].name, why, also), file=sys.stderr)
                        raise why
            else:
                done = usual()
            # A kernel that requires its block size (`required_threads`) is launched with that block
            # and no other: a translation with more registers than such a block may have on a V100
            # never runs, and a program that reads the limit from the loaded kernel (Triton) stops
            # before any launch, so that the second translation at launch never comes. The first of
            # levels 1, 2 and 3 that keeps within it is taken instead; when none does, the kernel is
            # refused for its registers, and the layer runs the PTX that the cubin keeps, when it
            # keeps some. (The fewest registers that the block of a kernel of the group allows.)
            required = []
            for f in fs:
                info = e.section('.nv.info.' + f.name)
                threads = required_threads(info.data) if info is not None else None
                if threads is not None:
                    required.append((block_registers(threads), threads))
            if required and max(r[2] for r in done.values()) > min(required)[0]:
                limit, threads = min(required)
                fitted = None
                if BORROW and alone:
                    top = max(0, limit - 3 - fs[0].save_slots)
                    for bounded in (lambda: attempt(True, True, 1), lambda: attempt(True, True, 2),
                                    lambda: attempt(False, False, 3, top)):
                        try:
                            candidate = bounded()
                        except Refused:
                            continue
                        if max(r[2] for r in candidate.values()) <= limit:
                            fitted = candidate
                            break
                if fitted is None:
                    raise Refused('more registers than its required block of %d threads allows (%d)' % (threads, limit))
                done = fitted
            count = max(r[2] for r in done.values())
            if count > declared and not any(i in entries for i in indices):
                # No kernel of the file calls them: their callers are elsewhere (a library of
                # functions, linked later) and may keep something in the registers above those
                # these functions declare, which is where the translation would write. (Functions
                # that a kernel of the file calls share its registers: above.) Found on code the
                # translator had never seen.
                raise Refused('called from code that is not in the file, and the translation needs registers above its own')
            for i, f in zip(indices, fs):
                f.moved = f.follow([(offset, kind) for section, at, offset, kind in relocations.get(i, ())])
            for i, f in zip(indices, fs):
                info = e.section('.nv.info.' + f.name)
                if info is not None:
                    attributes(info.data, done[i][1], f.name)
                bank = e.section('.nv.constant2.' + f.name)
                for base, places in f.relocate:
                    if bank is None or base + 4 * len(places) > len(bank.data) or list(struct.unpack_from('<%dI' % len(places), bank.data, base)) != places:
                        raise Refused('a jump table is not in the constants where its load reads it')
            results.update({i: (done[i][0], done[i][1], count) for i in indices})
            # Track frame sizes for level 3 (memory slots).
            for i, f in zip(indices, fs):
                if f.borrow is not None and f.borrow.frame_bytes > 0:
                    results[i] = results[i] + (f.borrow.frame_bytes,)
                else:
                    results[i] = results[i] + (0,)
        except Refused as why:
            for f in fs:
                report[f.name] = str(why)
    # The container: code, sizes, register counts, attributes; refused functions become a trap.
    trap = mc.assemble('bpt__TRAP', Sa=1) if 'bpt__TRAP' in mc.md70.by_name else None
    counts, frames = {}, {}
    for i, s in index_of.items():
        name = s.name[len('.text.'):]
        if i in results:
            code, new_of, count = results[i][:3]
            frame_bytes = results[i][3] if len(results[i]) > 3 else 0
            if listing is not None:
                f = functions[i]
                for k, item in enumerate(f.items):
                    listing.append((name, 16 * k, 16 * item.origin, item.what))
            if lives is not None:
                f = functions[i]
                lives += [(name, q, p, saved, [16 * at for at in part]) for q, p, saved, part in f.lives]
                for q, part in f.held_lives:
                    if q in f.hold_at:
                        location, kind = f.hold_at[q], 'held'
                    else:
                        location, kind = -f.borrow.slot_offset(f.borrow.predicate_slots[q]), 'held-slot'
                    lives.append((name, q, location, kind, [16 * at for at in part]))
                if functions[i].borrow is not None:
                    # What was borrowed, for validate.py to check by its own lives: `home`, a
                    # uniform register (first number) in a register of the program (second)
                    # at the instructions of its life; `pool`, so many registers (first number)
                    # from this one (second) taken at one instruction (the pool, and at level 3
                    # whatever else the instruction was given). At level 3 also `slot`, a uniform
                    # register (first number) so many bytes under the top of the thread's local
                    # memory (second), and `frame`, the bytes added there (first number). (A
                    # `held` line whose register is one of the program's is borrowed too.)
                    b = functions[i].borrow
                    lives += [(name, u, r, 'home', [16 * at for at in b.spans[u]]) for u, r in sorted(b.homes.items())]
                    lives += [(name, width, first, 'pool', [16 * at]) for at, runs in sorted(functions[i].given_at.items()) for first, width in runs]
                    # (`room`: a register of the program (second number) put in the frame for
                    # the time of one instruction's sequence, and brought back.)
                    lives += [(name, 1, r, 'room', [16 * at]) for at, registers in sorted(functions[i].spilled.items()) for r in registers]
                    lives += [(name, u, -b.slot_offset(word), 'slot', []) for u, word in sorted(b.slots.items())]
                    if b.frame_bytes:
                        lives.append((name, b.frame_bytes, 0, 'frame', []))
            s.data = code
            s.info = (count << 24) | (s.info & 0xffffff)
            counts[s.info & 0xffffff] = count
            if frame_bytes > 0:
                frames[s.info & 0xffffff] = frame_bytes
            info = e.section('.nv.info.' + name)
            if info is not None:
                info.data = attributes(info.data, new_of, name)
            for base, places in functions[i].relocate:
                # A jump table: its places as they were, to where they are now.
                bank = e.section('.nv.constant2.' + name)
                moved = bytearray(bank.data)
                struct.pack_into('<%dI' % len(places), moved, base, *[new_of[t] for t in places])
                bank.data = bytes(moved)
            moved = {}
            for (section, at, offset, kind), now in zip(relocations.get(i, ()), functions[i].moved):
                # A relocation: to where its instruction is now.
                struct.pack_into('<Q', moved.setdefault(section, bytearray(section.data)), at, now)
            for section, entries_now in moved.items():
                section.data = bytes(entries_now)
            for sym in symbols:
                if sym[3] == i:
                    if sym[4] in new_of:
                        sym[4] = new_of[sym[4]]
                    elif sym[4]:
                        report[name] = 'a symbol at 0x%x, which is not an instruction' % sym[4]
                    if sym[1] & 0xf == 2:
                        sym[5] = len(code) - (sym[4] if arch == 80 else 0)
        else:
            # Never to run: the proxy refuses its kernels. What reads the file must still find valid
            # sm_70 code that agrees with its attributes (the driver visits the instructions they
            # list): instructions that do nothing, an EXIT wherever one is recorded, and no other
            # record of an instruction. Should it be launched all the same, it must not pass for a
            # kernel that ran: it starts with a trap, as the compiler writes one.
            info = e.section('.nv.info.' + name)
            exits = []
            if info is not None:
                pos = 0
                while pos + 4 <= len(info.data):
                    fmt, attr, val = info.data[pos], info.data[pos + 1], struct.unpack_from('<H', info.data, pos + 2)[0]
                    if fmt == 4 and attr == 0x1c:
                        whole = max(0, (min(pos + 4 + val, len(info.data)) - (pos + 4)) // 4)
                        exits = list(struct.unpack_from('<%dI' % whole, info.data, pos + 4))
                    pos += 4 + val if fmt == 4 else 4
                info.data = without(info.data, (OFFSET_LISTS - {0x1c}) | set(OFFSET_PAIRS) | {0x34})
            code = bytearray(NOP.to_bytes(16, 'little') * (len(s.data) // 16))
            leave = mc.assemble('exit_')
            leave = with_stall(leave, 5) | (16 << USCHED)
            for at in exits or [0]:
                if at + 16 <= len(code):
                    code[at:at + 16] = leave.to_bytes(16, 'little')
            for at, word in ((0, TRAP), (16, STAY)):
                if at + 16 <= len(code) and at not in exits:
                    code[at:at + 16] = word.to_bytes(16, 'little')
            s.data = bytes(code)
    e.set_symbols(symbols)
    info, raised = e.section('.nv.info'), set()
    if info is not None:
        d, pos = bytearray(info.data), 0
        while pos + 4 <= len(d):
            fmt, attr, val = d[pos], d[pos + 1], struct.unpack_from('<H', d, pos + 2)[0]
            if fmt == 4 and attr == 0x2f and pos + 12 <= len(d):
                symbol = struct.unpack_from('<I', d, pos + 4)[0]
                if symbol in counts:
                    struct.pack_into('<I', d, pos + 8, counts[symbol])
            # The frame of a function, the least stack it needs, and the most where that is
            # given: more by the frame the translation added (level 3). The driver sizes the
            # local memory of a thread by them.
            if fmt == 4 and attr in (0x11, 0x12, 0x23) and pos + 12 <= len(d):
                symbol = struct.unpack_from('<I', d, pos + 4)[0]
                if symbol in frames:
                    old = struct.unpack_from('<I', d, pos + 8)[0]
                    struct.pack_into('<I', d, pos + 8, old + frames[symbol])
                    raised.add((symbol, attr))
            pos += 4 + val if fmt == 4 else 4
        info.data = bytes(d)
    for symbol in frames:
        if not {(symbol, 0x11), (symbol, 0x12)} <= raised:
            raise ValueError('a frame was added to a function that has no frame attributes to raise')
    # The file is machine code for sm_70, in the format the driver's CUDA 12 side knows. It still
    # says that it was compiled for the architecture 7.5 (what `cudaFuncGetAttributes` gives as
    # `ptxVersion`): a program chooses by it among what it was built with (the algorithms of
    # Thrust and CUB answer "invalid device function" to an architecture below the lowest of
    # theirs), and it was built for nothing below.
    if e.header[7] == 0x41:
        e.header[7], e.header[8] = 0x33, 7
        e.header[0x14:0x18] = struct.pack('<I', 0x80)
        e.header[0x30:0x34] = struct.pack('<I', (arch << 16) | 0x0500 | 70)
        for s in e.sections:
            if s.name in ('.note.nv.tkinfo', '.note.nv.cuinfo'):
                s.type, s.data, s.size = 0, b'', 0
            elif s.name.startswith('.nv.info'):
                s.data = without(s.data, (0x5f, 0x66))
    else:
        e.header[0x30] = 70
    return e.write(), report


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    listing, lives = ([], []) if len(sys.argv) > 3 else (None, None)
    out, report = translate(open(sys.argv[1], 'rb').read(), listing=listing, lives=lives)
    open(sys.argv[2], 'wb').write(out)
    if listing is not None:
        with open(sys.argv[3], 'w') as f:
            for row in listing:
                f.write('%s\t%x\t%x\t%s\n' % row)
        write_lives(sys.argv[3] + '.lives', lives)
    refused = {k: v for k, v in report.items() if v}
    print('%s: %d functions, %d refused' % (os.path.basename(sys.argv[1]), len(report), len(refused)))
    for name, why in sorted(refused.items()):
        print('  refused %s: %s' % (name[:60], why))
