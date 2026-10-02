"""生成懒人包里的模型补丁：dst = 把 ops 作用在 src 上。

ops.txt 每行一条：
  C <src 文件偏移> <长度>   从原版文件复制
  L <lit.bin 偏移> <长度>   从 lit.bin 复制（新数据，也包括文件头和目录）
启动器（oneclick/launcher/launch.ps1 的 Apply-ModelPatch）按顺序执行这些操作，纯 PowerShell，用户不用装 Python。

用法（在 ninfer 源码根目录运行，要用到 tools.artifact）：
  python -X utf8 mkpatch.py swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer out_dir
Swift 1.5 IQ2_S → MTP Q4：只有 7 个 MTP 张量不同，310 条，复制 9.95 GB，新数据 204 MB。
"""
import argparse, os
from tools.artifact.reader import Artifact

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('out')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
A = Artifact.open(a.src); B = Artifact.open(a.dst)
assert len(A.directory.files) == 1 and len(B.directory.files) == 1, 'only single-file artifacts'
sobj = {o.id: o for o in A.objects}
ops = []

def add(kind, off, n):
    if n <= 0: return
    if ops and ops[-1][0] == kind and ops[-1][1] + ops[-1][2] == off:
        ops[-1] = (kind, ops[-1][1], ops[-1][2] + n); return
    ops.append((kind, off, n))

def edge(art, o):
    k = min(o.bytes, 1 << 16)
    return art.read_range(o.offset, k) + art.read_range(o.offset + o.bytes - k, k)

cur = 0; changed = []
for o in sorted(B.objects, key=lambda x: x.offset):
    st = B.payload_offset + o.offset
    assert st >= cur
    if st > cur: add('L', cur, st - cur)
    s = sobj.get(o.id)
    if s is not None and s.bytes == o.bytes and edge(A, s) == edge(B, o): add('C', A.payload_offset + s.offset, o.bytes)
    else: add('L', st, o.bytes); changed.append(o.id)
    cur = st + o.bytes
if B.file_bytes > cur: add('L', cur, B.file_bytes - cur)
print('changed objects:', len(changed), changed[:20])

fd = open(a.dst, 'rb'); lit = open(os.path.join(a.out, 'lit.bin'), 'wb'); lines = []; lo = 0
for k, off, n in ops:
    if k == 'C': lines.append('C %d %d' % (off, n))
    else:
        fd.seek(off); left = n
        while left:
            b = fd.read(min(left, 64 << 20)); lit.write(b); left -= len(b)
        lines.append('L %d %d' % (lo, n)); lo += n
lit.close()
open(os.path.join(a.out, 'ops.txt'), 'w', newline='\n').write('\n'.join(lines) + '\n')
print('ops', len(lines), 'lit bytes', lo, 'dst bytes', B.file_bytes, 'copy bytes', sum(n for k, _, n in ops if k == 'C'))
print('启动器转换完会校验 SHA256；自己验证可以按 ops 复原后和 dst 逐字节比较。')
