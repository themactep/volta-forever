#!/usr/bin/env python3
"""The container of machine code (a cubin: an ELF64 file) read into sections and written back.

Written back, a file whose sections were not changed is the original byte for byte (`elf.py FILE...`
checks that); sections whose size changed move what follows them.
"""
import struct
import sys

SHT_NULL, SHT_SYMTAB, SHT_RELA, SHT_NOBITS, SHT_REL = 0, 2, 4, 8, 9
PT_LOAD, PT_PHDR = 1, 6


class Section:
    __slots__ = ('name', 'name_offset', 'type', 'flags', 'addr', 'offset', 'size', 'link', 'info', 'align', 'entsize', 'data')

    def __repr__(self):
        return 'Section(%s)' % self.name


class Elf:
    def __init__(self, data):
        self.original = data
        assert data[:4] == b'\x7fELF' and data[4] == 2 and data[5] == 1, 'not a 64-bit little-endian ELF'
        self.header = bytearray(data[:64])
        (self.phoff, self.shoff) = struct.unpack_from('<QQ', data, 0x20)
        (self.phentsize, self.phnum, self.shentsize, self.shnum, self.shstrndx) = struct.unpack_from('<HHHHH', data, 0x36)
        assert self.shentsize == 64 and (self.phnum == 0 or self.phentsize == 56)
        assert self.shoff + 64 * self.shnum <= len(data) and self.phoff + 56 * self.phnum <= len(data), 'headers outside the file'
        self.sections = []
        for i in range(self.shnum):
            s = Section()
            (s.name_offset, s.type, s.flags, s.addr, s.offset, s.size, s.link, s.info, s.align, s.entsize) = struct.unpack_from('<IIQQQQIIQQ', data, self.shoff + 64 * i)
            assert s.type in (SHT_NULL, SHT_NOBITS) or s.offset + s.size <= len(data), 'a section outside the file'
            # No section of a cubin asks for more than 128.
            assert s.align <= 1 << 16, 'a section with an alignment of %d' % s.align
            s.data = b'' if s.type in (SHT_NULL, SHT_NOBITS) else data[s.offset:s.offset + s.size]
            self.sections.append(s)
        names = self.sections[self.shstrndx].data
        for s in self.sections:
            s.name = names[s.name_offset:names.index(b'\0', s.name_offset)].decode()
        self.segments = [list(struct.unpack_from('<IIQQQQQQ', data, self.phoff + 56 * i)) for i in range(self.phnum)]
        # (type, flags, offset, vaddr, paddr, filesz, memsz, align)

    def section(self, name):
        return next((s for s in self.sections if s.name == name), None)

    def index(self, section):
        return self.sections.index(section)

    # ---- symbols and relocations ---------------------------------------------------------------------
    def symbols(self):
        """[name, info, other, shndx, value, size] for each symbol; write back with set_symbols."""
        table = next(s for s in self.sections if s.type == SHT_SYMTAB)
        strings = self.sections[table.link].data
        out = []
        assert len(table.data) % 24 == 0, 'a table of symbols cut short'
        for i in range(0, len(table.data), 24):
            name, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', table.data, i)
            out.append([strings[name:strings.index(b'\0', name)].decode(), info, other, shndx, value, size, name])
        return out

    def set_symbols(self, symbols):
        table = next(s for s in self.sections if s.type == SHT_SYMTAB)
        table.data = b''.join(struct.pack('<IBBHQQ', s[6], s[1], s[2], s[3], s[4], s[5]) for s in symbols)

    # ---- writing ----------------------------------------------------------------------------------------
    def write(self):
        out = bytearray(self.header)
        offsets = []
        for s in self.sections:
            if s.type == SHT_NULL:
                offsets.append(0)
                continue
            align = max(1, s.align)
            out += b'\0' * (-len(out) % align)
            offsets.append(len(out))
            if s.type != SHT_NOBITS:
                out += s.data
        out += b'\0' * (-len(out) % 8)
        shoff = len(out)
        for s, offset in zip(self.sections, offsets):
            size = s.size if s.type in (SHT_NULL, SHT_NOBITS) else len(s.data)
            out += struct.pack('<IIQQQQIIQQ', s.name_offset, s.type, s.flags, s.addr, offset, size, s.link, s.info, s.align, s.entsize)
        phoff = len(out) if self.phnum else 0
        for seg in self.segments:
            type_, flags, offset, vaddr, paddr, filesz, memsz, align = seg
            if type_ == PT_PHDR or (type_ == PT_LOAD and offset == self.phoff):
                offset, filesz, memsz = phoff, 56 * self.phnum, 56 * self.phnum
            elif filesz or memsz:
                # The sections the segment held: it now holds them where they are, up to the end of the
                # last one (a section without contents counts by its place, which is aligned).
                def length(s):
                    return 0 if s.type == SHT_NOBITS else s.size
                inside = [i for i, s in enumerate(self.sections)
                          if s.type != SHT_NULL and s.offset >= offset and s.offset + length(s) <= offset + filesz]
                if inside:
                    start = offsets[inside[0]]
                    end = max(offsets[i] + (0 if self.sections[i].type == SHT_NOBITS else len(self.sections[i].data)) for i in inside)
                    offset, memsz, filesz = start, (memsz - filesz + (end - start)) % (1 << 64), end - start
            out += struct.pack('<IIQQQQQQ', type_, flags, offset, vaddr, paddr, filesz, memsz, align)
        struct.pack_into('<QQ', out, 0x20, phoff, shoff)
        return bytes(out)


if __name__ == '__main__':
    same = 0
    for path in sys.argv[1:]:
        data = open(path, 'rb').read()
        again = Elf(data).write()
        if again == data:
            same += 1
        else:
            first = next((i for i in range(min(len(data), len(again))) if data[i] != again[i]), None)
            print('%s: written back differently (lengths %d and %d, first difference at 0x%x)' % (path, len(data), len(again), first if first is not None else -1))
    print('%d of %d files written back byte for byte' % (same, len(sys.argv) - 1))
