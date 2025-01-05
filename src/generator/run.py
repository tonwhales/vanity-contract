import hashlib
from threading import Thread
import pyopencl as cl
import numpy as np
import base64
import time
import os
import argparse

parser = argparse.ArgumentParser('vanity-generator', description='Generate beautiful wallet for TON on GPU using vanity contract')
parser.add_argument('owner', help='An owner of vanity contract')
parser.add_argument('--end', type=str, default='', help='Search in the end of address, separated by , or |')
parser.add_argument('--start', type=str, default='', help='Search in the start of address, separated by , or |')
parser.add_argument('-nb', action='store_true', default=False, help='Search for non-bouncable addresses')
parser.add_argument('-t', action='store_true', default=False, help='Search for testnet addresses')
parser.add_argument('--threads', type=int, help='Worker threads')
parser.add_argument('--its', type=int, default=10000, help='Worker iterations')
parser.add_argument('-w', type=int, required=True, help='Address workchain')
parser.add_argument('--case-sensitive', action='store_true', help='Search for case sensitive address (case insensitive by default)')
parser.add_argument('--early-prefix', action='store_true', help='Check prefix starting from third character (subject to some address format restrictions, fourth by default)')
parser.add_argument('--only-one', action='store_true', help='Stop when an address is found (runs until interrupted by default)')

args = parser.parse_args()
if not args.end and not args.start:
    parser.print_usage()
    print('vanity-generator: error: the following arguments are required: end or start')
    os._exit(0)

def parse_parameters(param):
    return [p.strip() for p in param.replace('|', ',').split(',') if p.strip()]

start_patterns = parse_parameters(args.start)
end_patterns = parse_parameters(args.end)

OWNER = args.owner
owner_decoded = base64.urlsafe_b64decode(OWNER)
inner_base = bytearray.fromhex('00840400') + owner_decoded[2:34]

BOUNCEABLE_TAG = 0x11
NON_BOUNCEABLE_TAG = 0x51
TEST_FLAG = 0x80
WORKCHAIN = (args.w + (1 << 8)) % (1 << 8)

flags = (NON_BOUNCEABLE_TAG if args.nb else BOUNCEABLE_TAG)
if args.t:
    flags |= TEST_FLAG
flags <<= 8
flags |= WORKCHAIN

if not args.case_sensitive:
    start_patterns = [p.lower() for p in start_patterns]
    end_patterns = [p.lower() for p in end_patterns]

start_offset = 3

conditions = []
start_conditions = []
end_conditions = []
if args.early_prefix:
    conditions.append('early-prefix')
    start_offset = 2
if args.case_sensitive:
    conditions.append('case-sensitive')
if start_patterns:
    conditions.append(f"starting with {start_patterns}")
    start_conditions.append(
        " || ".join(
            [
                "("
                + " && ".join(
                    [
                        (
                            f"result[{i + start_offset}] == '{c}'"
                            if args.case_sensitive
                            else f"(result[{i + start_offset}] == '{c}' || result[{i + start_offset}] == '{c.upper()}')"
                        )
                        for i, c in enumerate(pattern)
                    ]
                )
                + ")"
                for pattern in start_patterns
            ]
        )
    )
if end_patterns:
    conditions.append(f"with {end_patterns} in the end")
    end_conditions.append(
        " || ".join(
            [
                "("
                + " && ".join(
                    [
                        (
                            f"result[{47 - len(pattern) + i + 1}] == '{c}'"
                            if args.case_sensitive
                            else f"(result[{47 - len(pattern) + i + 1}] == '{c}' || result[{47 - len(pattern) + i + 1}] == '{c.upper()}')"
                        )
                        for i, c in enumerate(pattern)
                    ]
                )
                + ")"
                for pattern in end_patterns
            ]
        )
    )
kernel_conditions = []
if start_conditions and end_conditions:
    kernel_conditions.append(f'(({" && ".join(start_conditions)}) && ({" && ".join(end_conditions)}))')
elif start_conditions:
    kernel_conditions.append(f'({" && ".join(start_conditions)})')
elif end_conditions:
    kernel_conditions.append(f'({" && ".join(end_conditions)})')


print()
print('Searching wallets', ', '.join(conditions))
print("Owner: ", OWNER)
print("Flags: ", flags.to_bytes(2, 'big').hex())
print("Kernel conditions:", ' && '.join(kernel_conditions))
print()

mf = cl.mem_flags
n_found = 0

def crc16(data):
    reg = 0
    for b in data:
        mask = 0x80
        while mask > 0:
            reg <<= 1
            if b & mask:
                reg = reg + 1
            mask >>= 1
            if reg > 0xffff:
                reg = reg & 0xffff
                reg ^= 0x1021
    return reg.to_bytes(2, byteorder='big')

def solver(dev, context, queue, program):
    global n_found
    main = bytearray.fromhex('020134000100009b598624c569108630d69c8422af4b5971cd9d515ad83d4facec29e25b2f9c75d7c2f9ece11a5845e257cc6c8bd375459059902ce9f6206696a8964c5e7e078100')
    data = np.frombuffer(main, dtype=np.uint32)
    main_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 72, hostbuf=data)

    salt = bytearray(os.urandom(32))
    inner = inner_base + salt
    inner_data = np.frombuffer(inner, dtype=np.uint32)
    inner_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 68, hostbuf=inner_data)

    res = np.full(2048, 0xffffffff, np.uint32)
    res_g = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

    start = time.time()
    threads = args.threads or (dev.max_work_group_size * dev.max_compute_units)
    iterations = args.its
    program.hash_main(
        queue,
        (threads,),
        None,
        np.int32(iterations),
        np.ushort(flags),
        np.int32(71),
        main_g,
        np.int32(68),
        inner_g,
        res_g
    ).wait()
    result = np.empty(2048, np.uint32)
    cl.enqueue_copy(queue, result, res_g).wait()

    ps = list(np.where(result != 0xffffffff))[0]
    misses = 0
    if len(ps):
        for j in range(0, len(ps), 2):
            p = ps[j]
            assert ps[j + 1] == p + 1
            a = result[p]
            b = result[p+1]

            salt_np = np.frombuffer(salt, np.uint32)
            salt_np[0] ^= a
            salt_np[1] ^= b
            hdata1 = inner_base + salt_np.tobytes()
            hash1 = hashlib.sha256(hdata1).digest()
            main[39:71] = hash1

            hs = hashlib.sha256(main[:71]).digest()

            address = bytearray()
            address += flags.to_bytes(2, 'big') # flags
            address += hs
            address += b'\x00\x00'
            crc = crc16(address)
            address[34] = crc[0]
            address[35] = crc[1]
            found = base64.urlsafe_b64encode(address).decode('utf-8')

            if any(found.lower().endswith(pattern) for pattern in end_patterns) or \
               any(found[start_offset:].lower().startswith(pattern) for pattern in start_patterns):
                print('Found: ', found, 'salt: ', salt_np.tobytes().hex())
                with open('found.txt', 'a') as f:
                    f.write(f'{found} {salt_np.tobytes().hex()}\n')
                if args.only_one:
                    stopped = True
                    os._exit(0)
                n_found += 1
            else:
                misses += 1
    print('Speed:', round(threads * iterations / (time.time() - start) / 1e6), 'Mh/s, miss: ' + str(misses) + ', found: ' + str(n_found))

kernel_code = open(os.path.join(os.path.dirname(__file__), 'vanity.cl')).read()
kernel_code = kernel_code.replace("<<CONDITION>>", ' && '.join(kernel_conditions))

stopped = False
def device_thread(device):
    context = cl.Context(devices=[device], dev_type=None)
    queue = cl.CommandQueue(context)
    program = cl.Program(context, kernel_code).build()

    while not stopped:
        solver(device, context, queue, program)
    pass

platforms = cl.get_platforms()
dev_n = 1
threads = []
for platform in platforms:
    devices = platform.get_devices(cl.device_type.GPU)
    for dev in devices:
        print("Using device: ", dev.name)
        t = Thread(None, device_thread, 'dev-{}'.format(dev_n), (dev,))
        threads.append(t)
        t.start()
        dev_n += 1

try:
    [t.join(1) for t in threads]
except KeyboardInterrupt:
    print('Interrupted')
    stopped = True
    os._exit(0)
