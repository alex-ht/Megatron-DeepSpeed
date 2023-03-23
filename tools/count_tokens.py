import sys
from megatron.data import indexed_dataset


src = sys.argv[1]
builder = indexed_dataset.make_dataset(src, 'mmap')
print("#tokens:", sum(builder.sizes))